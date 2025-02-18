using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using Accord.Math.Optimization;
using Warp.Headers;
using Warp.Sociology;
using Warp.Tools;

namespace Warp;

public partial class Movie
{
    public virtual void PerformMultiParticleRefinement(
        string workingDirectory,
        ProcessingOptionsMPARefine optionsMPA,
        Species[] allSpecies,
        DataSource dataSource,
        Image gainRef,
        DefectModel defectMap,
        Action<string> progressCallback
    )
    {
        int GPUID = GPU.GetDevice();
        HeaderEER.GroupNFrames = dataSource.EERGroupFrames;

        NFrames = Math.Min(MapHeader.ReadFromFile(DataPath).Dimensions.Z, dataSource.FrameLimit);
        //NFrames = 1;
        FractionFrames = (float)NFrames / MapHeader.ReadFromFile(DataPath).Dimensions.Z;
        int RedNFrames = NFrames;

        float BfactorWeightingThreshold = (float)optionsMPA.BFactorWeightingThreshold;

        //MagnificationCorrection = new float3(1, 1, 0);
        //CTF.BeamTilt = new float2(0, 0);

        if (CTF.ZernikeCoeffsOdd == null)
            CTF.ZernikeCoeffsOdd = new float[12];
        else if (CTF.ZernikeCoeffsOdd.Length < 12)
            CTF.ZernikeCoeffsOdd = Helper.Combine(CTF.ZernikeCoeffsOdd, new float[12 - CTF.ZernikeCoeffsOdd.Length]);

        if (CTF.ZernikeCoeffsEven == null)
            CTF.ZernikeCoeffsEven = new float[8];
        else if (CTF.ZernikeCoeffsEven.Length < 8)
            CTF.ZernikeCoeffsEven = Helper.Combine(CTF.ZernikeCoeffsEven, new float[8 - CTF.ZernikeCoeffsEven.Length]);

        //GridLocationWeights = new CubicGrid(new int3(1), new float[] { 1f });
        //GridLocationBfacs = new CubicGrid(new int3(1), new float[] { 0f });

        //GridAngleX = new CubicGrid(new int3(1), new float[1]);
        //GridAngleY = new CubicGrid(new int3(1), new float[1]);
        //GridAngleZ = new CubicGrid(new int3(1), new float[1]);

        #region Get particles belonging to this item; if there are none, abort

        string DataHash = GetDataHash();

        Dictionary<Species, Particle[]> SpeciesParticles = new Dictionary<Species, Particle[]>();
        foreach (var species in allSpecies)
            SpeciesParticles.Add(species, species.GetParticles(DataHash));

        if (SpeciesParticles.Select(p => p.Value.Length).Sum() < optionsMPA.MinParticlesPerItem)
            return;

        //float2 ParticleCenter = MathHelper.Mean(SpeciesParticles[allSpecies[0]].Select(p => new float2(p.Coordinates[0].X, p.Coordinates[0].Y)));
        ////SpeciesParticles[allSpecies[0]] = SpeciesParticles[allSpecies[0]].Where(p => p.Coordinates[0].X >= ParticleCenter.X && p.Coordinates[0].Y <= ParticleCenter.Y).ToArray();
        //List<Particle> ParticlesSorted = SpeciesParticles[allSpecies[0]].ToList();
        //Dictionary<Particle, float> ParticleDistances = new Dictionary<Particle, float>();
        //foreach (var p in ParticlesSorted)
        //    ParticleDistances.Add(p, (ParticleCenter - new float2(p.Coordinates[0].X, p.Coordinates[0].Y)).Length());
        //ParticlesSorted.Sort((a, b) => ParticleDistances[a].CompareTo(ParticleDistances[b]));

        ////SpeciesParticles[allSpecies[0]] = ParticlesSorted.Skip(1).Take(1).ToArray();

        #endregion

        float4[] ParticleMags = Helper.ArrayOfConstant(new float4(0, 0, 0, 0), SpeciesParticles[allSpecies[0]].Length);

        #region Figure out dimensions

        {
            MapHeader Header = MapHeader.ReadFromFile(DataPath);
            ImageDimensionsPhysical = new float2(new int2(Header.Dimensions)) * (float)dataSource.PixelSizeMean;
            if (Header.GetType() == typeof(HeaderEER) && gainRef != null)
                ImageDimensionsPhysical = new float2(new int2(gainRef.Dims)) * (float)dataSource.PixelSizeMean;
        }

        float SmallestAngPix = MathHelper.Min(allSpecies.Select(s => (float)s.PixelSize));
        float LargestBox = MathHelper.Max(allSpecies.Select(s => s.DiameterAngstrom)) * 2 / SmallestAngPix;

        float[] DoseInterpolationSteps = Helper.ArrayOfFunction(i => (float)i / Math.Max(1, NFrames - 1), NFrames);

        #endregion

        #region Load and preprocess frames

        progressCallback("Loading movie frame data...");

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
                1),

            Invert = true,
            NormalizeInput = true,
            NormalizeOutput = false,

            PrerotateParticles = true
        };

        Image[] FrameData;
        LoadFrameData(OptionsDataLoad, gainRef, defectMap, out FrameData);

        float2 AverageMeanStd;
        {
            Image Average = new Image(FrameData[0].Dims);
            foreach (var frame in FrameData)
            {
                Average.Add(frame);
                frame.FreeDevice();
            }

            Average.Multiply(1f / FrameData.Length);

            float[] MeanPlaneData = MathHelper.FitAndGeneratePlane(Average.GetHost(Intent.Read)[0], new int2(Average.Dims));
            Image MeanPlane = new Image(MeanPlaneData, Average.Dims);
            Image PaddedFrame = new Image(IntPtr.Zero, Average.Dims + new int3(2048, 2048, 1));

            Average.Fill(0f);

            foreach (var frame in FrameData)
            {
                frame.Subtract(MeanPlane);

                GPU.PadClamped(frame.GetDevice(Intent.Read),
                    PaddedFrame.GetDevice(Intent.Write),
                    frame.Dims,
                    PaddedFrame.Dims,
                    1);

                PaddedFrame.Bandpass(1f / LargestBox, 1f, false, 0f);

                GPU.Pad(PaddedFrame.GetDevice(Intent.Read),
                    frame.GetDevice(Intent.Write),
                    PaddedFrame.Dims,
                    frame.Dims,
                    1);

                Average.Add(frame);

                frame.FreeDevice();
            }

            AverageMeanStd = MathHelper.MeanAndStd(Average.GetHost(Intent.Read)[0]);
            Average.Dispose();
            MeanPlane.Dispose();
            PaddedFrame.Dispose();
        }

        for (int z = 0; z < FrameData.Length; z++)
        {
            FrameData[z].Add(-AverageMeanStd.X);
            FrameData[z].Multiply(-1f / AverageMeanStd.Y);

            FrameData[z].FreeDevice();
        }

        if (false)
        {
            Image Average = new Image(FrameData[0].Dims);
            foreach (var frame in FrameData)
                Average.Add(frame);
            Average.Multiply(1f / FrameData.Length);

            if (GPUID == 0)
                Average.WriteMRC("d_avg.mrc", true);
            Average.Dispose();
        }

        #endregion

        #region Compose optimization steps based on user's requests

        var OptimizationStepsWarp = new List<(WarpOptimizationTypes Type, int Iterations, string Name)>();
        {
            OptimizationStepsWarp.Add((WarpOptimizationTypes.ParticleMag, 5, "particle magnification"));

            if (optionsMPA.DoMagnification)
            {
                OptimizationStepsWarp.Add((WarpOptimizationTypes.Magnification, 5, "magnification"));
                //OptimizationStepsWarp.Add((WarpOptimizationTypes.Magnification | WarpOptimizationTypes.ParticlePosition | WarpOptimizationTypes.ParticleAngle, 10, "magnification"));
            }

            {
                WarpOptimizationTypes TranslationComponents = 0;
                if (optionsMPA.DoImageWarp)
                    TranslationComponents |= WarpOptimizationTypes.ImageWarp;

                if (TranslationComponents != 0)
                    OptimizationStepsWarp.Add((TranslationComponents, 10, "image warping"));
            }
            {
                WarpOptimizationTypes AntisymComponents = 0;

                if (optionsMPA.DoZernike13)
                    AntisymComponents |= WarpOptimizationTypes.Zernike13;
                if (optionsMPA.DoZernike5)
                    AntisymComponents |= WarpOptimizationTypes.Zernike5;

                if (AntisymComponents != 0 && allSpecies.Any(s => s.ResolutionRefinement < (float)optionsMPA.MinimumCTFRefinementResolution))
                    OptimizationStepsWarp.Add((AntisymComponents | WarpOptimizationTypes.ParticlePosition, 20, "antisymmetrical aberrations"));
            }
            {
                if (optionsMPA.DoDoming)
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
        }


        var OptimizationStepsCTF = new List<(CTFOptimizationTypes Type, int Iterations, string Name)>();
        {
            //if (optionsMPA.DoMagnification)
            //    OptimizationStepsCTF.Add((CTFOptimizationTypes.Distortion, 5, "Distortion"));

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
            //if (optionsMPA.DoMagnification)
            //    DefocusComponents |= CTFOptimizationTypes.Distortion;

            if (optionsMPA.DoDoming)
                DefocusComponents |= CTFOptimizationTypes.Doming;

            if (DefocusComponents != 0)
                OptimizationStepsCTF.Add((DefocusComponents, 10, "CTF parameters"));

            if (optionsMPA.DoCs)
                OptimizationStepsCTF.Add((CTFOptimizationTypes.Cs, 10, "Cs"));

            CTFOptimizationTypes ZernikeComponents = 0;

            if (optionsMPA.DoZernike2)
                ZernikeComponents |= CTFOptimizationTypes.Zernike2;
            if (optionsMPA.DoZernike4)
                ZernikeComponents |= CTFOptimizationTypes.Zernike4;

            if (ZernikeComponents != 0)
                OptimizationStepsCTF.Add((ZernikeComponents, 10, "symmetrical aberrations"));
        }

        #endregion

        if (optionsMPA.NIterations > 0)
        {
            #region Resize grids

            int2 MovementSpatialDims = new int2(optionsMPA.ImageWarpWidth, optionsMPA.ImageWarpHeight);

            if (optionsMPA.DoImageWarp)
                if (PyramidShiftX.Count == 0 ||
                    PyramidShiftX[0].Dimensions.X != MovementSpatialDims.X ||
                    PyramidShiftX[0].Dimensions.Y != MovementSpatialDims.Y)
                {
                    PyramidShiftX.Clear();
                    PyramidShiftY.Clear();

                    float OverallDose = (float)(dataSource.DosePerAngstromFrame < 0 ? -dataSource.DosePerAngstromFrame : dataSource.DosePerAngstromFrame * NFrames);

                    int NTemporal = (int)Math.Min(MathF.Ceiling(OverallDose), NFrames);

                    while(true)
                    {
                        PyramidShiftX.Add(new CubicGrid(new int3(MovementSpatialDims.X, MovementSpatialDims.Y, NTemporal)));
                        PyramidShiftY.Add(new CubicGrid(new int3(MovementSpatialDims.X, MovementSpatialDims.Y, NTemporal)));

                        MovementSpatialDims *= 2;
                        NTemporal = (NTemporal + 3) / 4;
                        if (NTemporal < 3)
                            break;
                    }
                }

            int AngleSpatialDim = 3;

            if (optionsMPA.DoDoming)
                if (GridAngleX == null || GridAngleX.Dimensions.X < AngleSpatialDim || GridAngleX.Dimensions.Z != NFrames)
                {
                    GridAngleX = GridAngleX == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NFrames)) : GridAngleX.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NFrames));
                    GridAngleY = GridAngleY == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NFrames)) : GridAngleY.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NFrames));
                    GridAngleZ = GridAngleZ == null ? new CubicGrid(new int3(AngleSpatialDim, AngleSpatialDim, NFrames)) : GridAngleZ.Resize(new int3(AngleSpatialDim, AngleSpatialDim, NFrames));
                }

            int DomingSpatialDim = 3;
            if (GridCTFDoming == null || GridCTFDoming.Dimensions.X < DomingSpatialDim || GridCTFDoming.Dimensions.Z != NFrames)
            {
                GridCTFDoming = GridCTFDoming == null ? new CubicGrid(new int3(DomingSpatialDim, DomingSpatialDim, NFrames)) : GridCTFDoming.Resize(new int3(DomingSpatialDim, DomingSpatialDim, NFrames));
            }

            if (GridDoseBfacs != null && GridDoseBfacs.Values.Length != NFrames)
                GridDoseBfacs = new CubicGrid(new int3(1, 1, NFrames),
                    Helper.ArrayOfFunction(i => -(i + 0.5f) * (float)dataSource.DosePerAngstromFrame * 4, NFrames));

            //if (GridCTFDefocusDelta.Values.Length <= 1)
            {
                //GridCTFDefocusDelta = new CubicGrid(new int3(3, 3, 1), Helper.ArrayOfConstant((float)CTF.DefocusDelta, 9));
                //GridCTFDefocusAngle = new CubicGrid(new int3(3, 3, 1), Helper.ArrayOfConstant((float)CTF.DefocusAngle, 9));

                if (GridCTFDefocusDelta == null || (GridCTFDefocusDelta.Dimensions.Elements() == 1 && GridCTFDefocusDelta.Values[0] == 0))
                {
                    GridCTFDefocusDelta = new CubicGrid(new int3(3, 3, 1), Helper.ArrayOfConstant((float)CTF.DefocusDelta, 3 * 3));
                    GridCTFDefocusAngle = new CubicGrid(new int3(3, 3, 1), Helper.ArrayOfConstant((float)CTF.DefocusAngle, 3 * 3));
                }
                else
                {
                    GridCTFDefocusDelta = GridCTFDefocusDelta.Resize(new int3(3, 3, 1));
                    GridCTFDefocusAngle = GridCTFDefocusAngle.Resize(new int3(3, 3, 1));
                }
            }

            if (GridCTFCs == null || GridCTFCs.Dimensions.Elements() == 1)
            {
                Console.WriteLine("Initialized Cs grid");
                GridCTFCs = new CubicGrid(new int3(3, 3, 1));
            }

            #endregion

            #region Create species prerequisites and calculate spectral weights

            progressCallback("Extracting particles...");

            Dictionary<Species, IntPtr[]> SpeciesParticleImages = new Dictionary<Species, IntPtr[]>();
            Dictionary<Species, IntPtr[]> SpeciesParticleQImages = new Dictionary<Species, IntPtr[]>();
            Dictionary<Species, float[]> SpeciesParticleDefoci = new Dictionary<Species, float[]>();
            Dictionary<Species, float2[]> SpeciesParticleExtractedAt = new Dictionary<Species, float2[]>();
            Dictionary<Species, float[]> SpeciesParticleExtractedAtDefocus = new Dictionary<Species, float[]>();
            Dictionary<Species, Image> SpeciesFrameWeights = new Dictionary<Species, Image>();
            Dictionary<Species, Image> SpeciesCTFWeights = new Dictionary<Species, Image>();
            Dictionary<Species, IntPtr> SpeciesParticleSubsets = new Dictionary<Species, IntPtr>();
            Dictionary<Species, (int Start, int End)> SpeciesParticleIDRanges = new Dictionary<Species, (int Start, int End)>();
            Dictionary<Species, int> SpeciesRefinementSize = new Dictionary<Species, int>();
            Dictionary<Species, int[]> SpeciesRelevantRefinementSizes = new Dictionary<Species, int[]>();
            Dictionary<Species, int> SpeciesCTFSuperresFactor = new Dictionary<Species, int>();

            Dictionary<Species, Image> CurrentWeightsDict = SpeciesFrameWeights;

            float[] AverageSpectrum1DAll = new float[128];
            long[] AverageSpectrum1DAllSamples = new long[128];

            int NParticlesOverall = 0;

            Action AllocateParticleMemory = () =>
            {
                foreach (var species in allSpecies)
                {
                    if (SpeciesParticles[species].Length == 0)
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;

                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                    IntPtr[] ImagesFTPinned = Helper.ArrayOfFunction(t =>
                    {
                        long Footprint = (new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles;
                        if (optionsMPA.UseHostMemory)
                            return GPU.MallocHostPinned(Footprint);
                        else
                            return GPU.MallocDevice(Footprint);
                    }, RedNFrames);

                    SpeciesParticleImages[species] = ImagesFTPinned;

                    IntPtr[] ImagesFTQPinned = null;
                    if (species.DoEwald)
                    {
                        ImagesFTQPinned = Helper.ArrayOfFunction(t => GPU.MallocDevice((new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles), RedNFrames);

                        SpeciesParticleQImages[species] = ImagesFTQPinned;
                    }
                }
            };

            Action FreeParticleMemory = () =>
            {
                foreach (var species in allSpecies)
                {
                    if (SpeciesParticleImages.ContainsKey(species) && SpeciesParticleImages[species] != null)
                    {
                        if (optionsMPA.UseHostMemory)
                            foreach (var item in SpeciesParticleImages[species])
                                GPU.FreeHostPinned(item);
                        else
                            foreach (var item in SpeciesParticleImages[species])
                                GPU.FreeDevice(item);

                        SpeciesParticleImages[species] = null;

                        if (species.DoEwald)
                        {
                            foreach (var item in SpeciesParticleQImages[species])
                                GPU.FreeDevice(item);

                            SpeciesParticleQImages[species] = null;
                        }
                    }
                }
            };

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

                int[] PlanForw = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeFull, SizeFull, 1), (uint)NFrames), 8);

                int[] RelevantSizes = GetRelevantImageSizes(SizeFull, BfactorWeightingThreshold).Select(v => Math.Min(Size, v)).ToArray();

                #region Extract particle images

                Image AverageAmplitudes = new Image(new int3(SizeFull, SizeFull, 1), true);

                {
                    Image[] AverageAmplitudesThreads = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, 1), true), PlanForw.Length);

                    Image[] Images = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, NFrames)), PlanForw.Length);
                    Image[] ImagesFT = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, NFrames), true, true), PlanForw.Length);
                    Image[] ReducedFT = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, 1), true, true), PlanForw.Length);
                    Image[] ImagesAmplitudes = Helper.ArrayOfFunction(i => new Image(new int3(SizeFull, SizeFull, 1), true), PlanForw.Length);

                    GPU.CheckGPUExceptions();

                    Helper.ForCPU(0, NParticles, PlanForw.Length, (threadID) => GPU.SetDevice(GPUID), (p, threadID) =>
                    {
                        GetImagesForOneParticle(OptionsDataLoad,
                            FrameData,
                            SizeFull,
                            Particles[p].GetCoordinateSeries(DoseInterpolationSteps),
                            PlanForw[threadID],
                            ParticleDiameterPix,
                            16,
                            true,
                            Images[threadID],
                            ImagesFT[threadID]);

                        GPU.ReduceMean(ImagesFT[threadID].GetDevice(Intent.Read),
                            ReducedFT[threadID].GetDevice(Intent.Write),
                            (uint)ImagesFT[threadID].ElementsSliceReal,
                            (uint)ImagesFT[threadID].Dims.Z,
                            1);

                        GPU.Amplitudes(ReducedFT[threadID].GetDevice(Intent.Read),
                            ImagesAmplitudes[threadID].GetDevice(Intent.Write),
                            (uint)ReducedFT[threadID].ElementsComplex);

                        ImagesAmplitudes[threadID].Multiply(ImagesAmplitudes[threadID]);

                        AverageAmplitudesThreads[threadID].Add(ImagesAmplitudes[threadID]);
                    }, null);

                    for (int i = 0; i < PlanForw.Length; i++)
                    {
                        AverageAmplitudes.Add(AverageAmplitudesThreads[i]);

                        AverageAmplitudesThreads[i].Dispose();
                        Images[i].Dispose();
                        ImagesFT[i].Dispose();
                        ReducedFT[i].Dispose();
                        ImagesAmplitudes[i].Dispose();
                    }
                }

                for (int i = 0; i < PlanForw.Length; i++)
                    GPU.DestroyFFTPlan(PlanForw[i]);

                #endregion

                #region Calculate spectrum

                AverageAmplitudes.Multiply(1f / NParticles);
                //if (GPUID == 0)
                //    AverageAmplitudes.WriteMRC("d_avgamps.mrc", true);

                float[] Amps1D = new float[Size / 2];
                float[] Samples1D = new float[Size / 2];
                float[][] Amps2D = AverageAmplitudes.GetHost(Intent.Read);

                Helper.ForEachElementFT(new int2(SizeFull), (x, y, xx, yy, r, angle) =>
                {
                    //int idx = (int)Math.Round(r);
                    //if (idx < Size / 2)
                    //{
                    //    Amps1D[idx] += Amps2D[0][y * (Size / 2 + 1) + x];
                    //    Samples1D[idx]++;
                    //}
                    int idx = (int)Math.Round(r / (SizeFull / 2) * AverageSpectrum1DAll.Length);
                    if (idx < AverageSpectrum1DAll.Length)
                    {
                        AverageSpectrum1DAll[idx] += Amps2D[0][y * (SizeFull / 2 + 1) + x] * NParticles;
                        AverageSpectrum1DAllSamples[idx] += NParticles;
                    }
                });

                //for (int i = 0; i < Amps1D.Length; i++)
                //    Amps1D[i] = Amps1D[i] / Samples1D[i];

                //float Amps1DMean = MathHelper.Mean(Amps1D);
                //for (int i = 0; i < Amps1D.Length; i++)
                //    Amps1D[i] = Amps1D[i] / Amps1DMean;

                AverageAmplitudes.Dispose();

                #endregion

                #region Defoci and extraction positions

                float[] Defoci = new float[NParticles * NFrames];
                float2[] ExtractedAt = new float2[NParticles * NFrames];
                float[] ExtractedAtDefocus = new float[NParticles * NFrames];

                for (int p = 0; p < NParticles; p++)
                {
                    float3[] Positions = GetPositionInAllFrames(Particles[p].GetCoordinateSeries(DoseInterpolationSteps));
                    for (int f = 0; f < NFrames; f++)
                    {
                        Defoci[p * NFrames + f] = Positions[f].Z;
                        ExtractedAt[p * NFrames + f] = new float2(Positions[f].X, Positions[f].Y);
                        ExtractedAtDefocus[p * NFrames + f] = Positions[f].Z;
                    }
                }

                #endregion

                #region Subset indices

                int[] Subsets = Particles.Select(p => p.RandomSubset).ToArray();
                IntPtr SubsetsPtr = GPU.MallocDeviceFromHostInt(Subsets, Subsets.Length);

                #endregion

                #region CTF superres factor

                CTF MaxDefocusCTF = CTF.GetCopy();
                int MinimumBoxSize = Math.Max(species.HalfMap1Projector[GPUID].Dims.X, MaxDefocusCTF.GetAliasingFreeSize(species.ResolutionRefinement, ParticleDiameterPix));
                float CTFSuperresFactor = (float)Math.Ceiling((float)MinimumBoxSize / species.HalfMap1Projector[GPUID].Dims.X) + 1e-3f;

                #endregion

                SpeciesParticleImages.Add(species, null);
                if (species.DoEwald)
                    SpeciesParticleQImages.Add(species, null);
                SpeciesParticleDefoci.Add(species, Defoci);
                SpeciesParticleExtractedAt.Add(species, ExtractedAt);
                SpeciesParticleExtractedAtDefocus.Add(species, ExtractedAtDefocus);
                SpeciesParticleSubsets.Add(species, SubsetsPtr);
                SpeciesRefinementSize.Add(species, Size);
                SpeciesRelevantRefinementSizes.Add(species, RelevantSizes);
                SpeciesCTFSuperresFactor.Add(species, (int)CTFSuperresFactor);

                species.HalfMap1Projector[GPUID].PutTexturesOnDevice();
                species.HalfMap2Projector[GPUID].PutTexturesOnDevice();
            }

            #region Calculate 1D PS averaged over all species and particles

            {
                for (int i = 0; i < AverageSpectrum1DAll.Length; i++)
                    AverageSpectrum1DAll[i] /= Math.Max(1, AverageSpectrum1DAllSamples[i]);

                float SpectrumMean = MathHelper.Mean(AverageSpectrum1DAll);
                for (int i = 0; i < AverageSpectrum1DAll.Length; i++)
                    AverageSpectrum1DAll[i] /= SpectrumMean;

                for (int i = 0; i < AverageSpectrum1DAll.Length; i++)
                    if (AverageSpectrum1DAll[i] <= 0)
                    {
                        for (int j = 0; j < AverageSpectrum1DAll.Length; j++)
                        {
                            if (i - j >= 0 && AverageSpectrum1DAll[i - j] > 0)
                            {
                                AverageSpectrum1DAll[i] = AverageSpectrum1DAll[i - j];
                                break;
                            }

                            if (i + j < AverageSpectrum1DAll.Length && AverageSpectrum1DAll[i + j] > 0)
                            {
                                AverageSpectrum1DAll[i] = AverageSpectrum1DAll[i + j];
                                break;
                            }
                        }
                    }

                if (AverageSpectrum1DAll.Any(v => v <= 0))
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

                Image CTFCoords = CTF.GetCTFCoords(Size, Size, CTF.Distortion);
                Image Weights = GetCTFsForOneParticle(OptionsWeights, new float3(0.5f), CTFCoords, null, true, true);
                //Weights.Min(1);
                CTFCoords.Dispose();

                #endregion

                #region Divide weights by 1D PS, and create a 20 A high-passed version for CTF refinement

                float[][] WeightsData = Weights.GetHost(Intent.ReadWrite);
                for (int f = 0; f < NFrames; f++)
                    Helper.ForEachElementFT(new int2(Size), (x, y, xx, yy, r, angle) =>
                    {
                        if (r < Size / 2)
                        {
                            int idx = Math.Min(AverageSpectrum1DAll.Length - 1,
                                (int)Math.Round(r / (Size / 2) *
                                                (float)dataSource.PixelSizeMean /
                                                species.ResolutionRefinement *
                                                AverageSpectrum1DAll.Length));

                            WeightsData[f][y * (Size / 2 + 1) + x] /= AverageSpectrum1DAll[idx];
                        }
                        else
                        {
                            WeightsData[f][y * (Size / 2 + 1) + x] = 0;
                        }
                    });

                //Weights.FreeDevice();
                //if (GPUID == 0)
                //    Weights.WriteMRC($"d_weights_{species.Name}.mrc", true);

                Image WeightsRelevantlySized = new Image(new int3(Size, Size, NFrames), true);
                for (int t = 0; t < NFrames; t++)
                    GPU.CropFTRealValued(Weights.GetDeviceSlice(t, Intent.Read),
                        WeightsRelevantlySized.GetDeviceSlice(t, Intent.Write),
                        Weights.Dims.Slice(),
                        new int3(RelevantSizes[t]).Slice(),
                        1);
                //if (GPUID == 0)
                //    WeightsRelevantlySized.WriteMRC($"d_weightsrelevant_{species.Name}.mrc", true);
                Weights.Dispose();

                Image CTFWeights = WeightsRelevantlySized.GetCopyGPU();
                float[][] CTFWeightsData = CTFWeights.GetHost(Intent.ReadWrite);
                for (int t = 0; t < CTFWeightsData.Length; t++)
                {
                    int RelevantSize = RelevantSizes[t];
                    float R20 = Size * (species.ResolutionRefinement / 2 / 10f);
                    Helper.ForEachElementFT(new int2(RelevantSize), (x, y, xx, yy, r, angle) =>
                    {
                        float Weight = 1 - Math.Max(0, Math.Min(1, R20 - r));
                        CTFWeightsData[t][y * (RelevantSize / 2 + 1) + x] *= Weight;
                    });
                }

                CTFWeights.FreeDevice();
                //if (GPUID == 0)
                //    CTFWeights.WriteMRC($"d_ctfweights_{species.Name}.mrc", true);

                #endregion

                SpeciesCTFWeights.Add(species, CTFWeights);
                SpeciesFrameWeights.Add(species, WeightsRelevantlySized);
            }

            #endregion

            // Remove original tilt image data from device, and dispose masks
            for (int f = 0; f < NFrames; f++)
                FrameData[f].FreeDevice();

            #endregion

            #region Helper functions

            Action<bool> ReextractPaddedParticles = (CorrectBeamTilt) =>
            {
                Stopwatch Watch = new Stopwatch();
                Watch.Start();

                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                foreach (var species in allSpecies)
                {
                    if (!SpeciesParticles.ContainsKey(species))
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    int SuperresFactor = SpeciesCTFSuperresFactor[species];
                    int BatchSize = optionsMPA.BatchSize;
                    BatchSize /= SuperresFactor * SuperresFactor;

                    int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];
                    int[] SizesRelevant = SpeciesRelevantRefinementSizes[species];

                    float AngPixRefine = species.ResolutionRefinement / 2;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                    float2[] ExtractedAt = SpeciesParticleExtractedAt[species];

                    Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);
                    Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefineSuper);
                    Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

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

                            for (int f = 0; f < RedNFrames; f++)
                            {
                                float3[] CoordinatesFrame = new float3[CurBatch];
                                float3[] AnglesFrame = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                    CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];

                                float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);

                                float3[] Defoci = new float3[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p].X = ImageCoords[p].Z;
                                    Defoci[p].Y = Astigmatism[p].X;
                                    Defoci[p].Z = Astigmatism[p].Y;
                                    ExtractedAt[(batchStart + p) * NFrames + f] = new float2(ImageCoords[p]);
                                }

                                GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    FrameData[f].Dims.Slice(),
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
                                    GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, PQReverse[iewald], ExtractedCTF, true);

                                    GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                        ExtractedCTF.GetDevice(Intent.Read),
                                        ExtractedFT.GetDevice(Intent.Write),
                                        ExtractedCTF.ElementsComplex,
                                        1);
                                }
                                else
                                {
                                    GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                    ParticleDiameterPix / 2f * 1.3f,
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
                                    ExtractedCroppedFTRelevantSize.GetDevice(Intent.Write),
                                    new int3(SizeRefine).Slice(),
                                    new int3(SizesRelevant[f]).Slice(),
                                    (uint)CurBatch);

                                GPU.CopyDeviceToHostPinned(ExtractedCroppedFTRelevantSize.GetDevice(Intent.Read),
                                    new IntPtr((long)PQStorage[iewald][f] + (new int3(SizesRelevant[f]).Slice().ElementsFFT()) * 2 * batchStart * sizeof(float)),
                                    (new int3(SizesRelevant[f]).Slice().ElementsFFT()) * 2 * CurBatch);
                            }
                        }
                    }

                    CoordsCTF.Dispose();
                    GammaCorrection.Dispose();
                    PhaseCorrection.Dispose();
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

                //foreach (var image in FrameData)
                //    image.FreeDevice();

                Watch.Stop();
                Console.WriteLine($"Extracted particles in {Watch.ElapsedMilliseconds}");
            };

            Func<float2[]> GetRawShifts = () =>
            {
                float2[] Result = new float2[NParticlesOverall * NFrames];

                foreach (var species in allSpecies)
                {
                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    float SpeciesAngPix = species.ResolutionRefinement / 2;
                    if (NParticles == 0)
                        continue;

                    int Offset = SpeciesParticleIDRanges[species].Start;

                    float3[] ParticlePositions = new float3[NParticles * NFrames];
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                        for (int f = 0; f < NFrames; f++)
                            ParticlePositions[p * NFrames + f] = Positions[f];
                    }

                    float3[] ParticlePositionsProjected = GetPositionInAllFrames(ParticlePositions);
                    float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                    for (int p = 0; p < NParticles; p++)
                    for (int f = 0; f < NFrames; f++)
                        Result[(Offset + p) * NFrames + f] = (new float2(ParticlePositionsProjected[p * NFrames + f]) - ParticleExtractedAt[p * NFrames + f]);
                }

                return Result;
            };

            Func<float2, Species, float[]> GetRawCCSpecies = (shiftBias, Species) =>
            {
                Particle[] Particles = SpeciesParticles[Species];

                int NParticles = Particles.Length;
                float AngPixRefine = Species.ResolutionRefinement / 2;

                if (NParticles == 0)
                    return new float[NParticles * NFrames * 3];

                float[] SpeciesResult = new float[NParticles * RedNFrames * 3];
                float[] SpeciesResultQ = new float[NParticles * RedNFrames * 3];

                float3[] ParticlePositions = new float3[NParticles * NFrames];
                float3[] ParticleAngles = new float3[NParticles * NFrames];
                for (int p = 0; p < NParticles; p++)
                {
                    float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);
                    float3[] Angles = Particles[p].GetAngleSeries(DoseInterpolationSteps);

                    for (int f = 0; f < NFrames; f++)
                    {
                        ParticlePositions[p * NFrames + f] = Positions[f];
                        ParticleAngles[p * NFrames + f] = Angles[f]; // * Helper.ToRad;
                    }
                }

                float3[] ParticlePositionsProjected = GetPositionInAllFrames(ParticlePositions);
                float3[] ParticleAnglesInFrames = GetParticleAngleInAllFrames(ParticlePositions, ParticleAngles); // ParticleAngles;

                float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[Species];
                float2[] ParticleShifts = new float2[NFrames * NParticles];
                for (int p = 0; p < NParticles; p++)
                for (int t = 0; t < NFrames; t++)
                    ParticleShifts[p * NFrames + t] = (new float2(ParticlePositionsProjected[p * NFrames + t]) - ParticleExtractedAt[p * NFrames + t] + shiftBias) / AngPixRefine;

                int[] RelevantSizes = SpeciesRelevantRefinementSizes[Species];

                int SizeRefine = Species.HalfMap1Projector[GPUID].Dims.X;
                int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;

                Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixRefine, SizeRefine);
                Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                for (int f = 0; f < NFrames; f++)
                    GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                        PhaseCorrectionAll.GetDeviceSlice(f, Intent.Write),
                        PhaseCorrection.Dims.Slice(),
                        new int3(RelevantSizes[f]).Slice(),
                        1);

                if (RedNFrames != NFrames)
                {
                    float2[] RedParticleShifts = new float2[RedNFrames * NParticles];
                    for (int p = 0; p < NParticles; p++)
                    for (int f = 0; f < RedNFrames; f++)
                        RedParticleShifts[p * RedNFrames + f] = ParticleShifts[p * NFrames + f];

                    float3[] RedParticleAngles = new float3[RedNFrames * NParticles];
                    for (int p = 0; p < NParticles; p++)
                    for (int f = 0; f < RedNFrames; f++)
                        RedParticleAngles[p * RedNFrames + f] = ParticleAnglesInFrames[p * NFrames + f];

                    ParticleShifts = RedParticleShifts;
                    ParticleAnglesInFrames = RedParticleAngles;
                }

                GPU.MultiParticleDiff(SpeciesResult,
                    SpeciesParticleImages[Species],
                    SpeciesRefinementSize[Species],
                    RelevantSizes,
                    Helper.ToInterleaved(ParticleShifts),
                    Helper.ToInterleaved(ParticleAnglesInFrames),
                    MagnificationCorrection.ToVec(),
                    (Species.ResolutionRefinement < 8 ? SpeciesCTFWeights : SpeciesFrameWeights)[Species].GetDevice(Intent.Read),
                    PhaseCorrectionAll.GetDevice(Intent.Read),
                    Species.DoEwald ? CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize) : 0,
                    Species.CurrentMaxShellRefinement,
                    new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                    new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                    Species.HalfMap1Projector[GPUID].Oversampling,
                    Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                    SpeciesParticleSubsets[Species],
                    NParticles,
                    RedNFrames);

                if (Species.DoEwald)
                    GPU.MultiParticleDiff(SpeciesResultQ,
                        SpeciesParticleQImages[Species],
                        SpeciesRefinementSize[Species],
                        RelevantSizes,
                        Helper.ToInterleaved(ParticleShifts),
                        Helper.ToInterleaved(ParticleAnglesInFrames),
                        MagnificationCorrection.ToVec(),
                        (Species.ResolutionRefinement < 8 ? SpeciesCTFWeights : SpeciesFrameWeights)[Species].GetDevice(Intent.Read),
                        PhaseCorrectionAll.GetDevice(Intent.Read),
                        -CTF.GetEwaldRadius(SizeFull, (float)Species.PixelSize),
                        Species.CurrentMaxShellRefinement,
                        new[] { Species.HalfMap1Projector[GPUID].t_DataRe, Species.HalfMap2Projector[GPUID].t_DataRe },
                        new[] { Species.HalfMap1Projector[GPUID].t_DataIm, Species.HalfMap2Projector[GPUID].t_DataIm },
                        Species.HalfMap1Projector[GPUID].Oversampling,
                        Species.HalfMap1Projector[GPUID].DimsOversampled.X,
                        SpeciesParticleSubsets[Species],
                        NParticles,
                        RedNFrames);

                GPU.CheckGPUExceptions();

                PhaseCorrection.Dispose();
                PhaseCorrectionAll.Dispose();

                if (Species.DoEwald)
                    for (int i = 0; i < SpeciesResult.Length; i++)
                        SpeciesResult[i] += SpeciesResultQ[i];

                if (RedNFrames != NFrames)
                {
                    float[] FullSpeciesResult = new float[NParticles * NFrames * 3];
                    for (int p = 0; p < NParticles; p++)
                    for (int f = 0; f < RedNFrames; f++)
                    {
                        FullSpeciesResult[(p * NFrames + f) * 3 + 0] = SpeciesResult[(p * RedNFrames + f) * 3 + 0];
                        FullSpeciesResult[(p * NFrames + f) * 3 + 1] = SpeciesResult[(p * RedNFrames + f) * 3 + 1];
                        FullSpeciesResult[(p * NFrames + f) * 3 + 2] = SpeciesResult[(p * RedNFrames + f) * 3 + 2];
                    }

                    SpeciesResult = FullSpeciesResult;
                }

                return SpeciesResult;
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

                    float[] SpeciesResult = GetRawCCSpecies(shiftBias, Species);

                    int Offset = SpeciesParticleIDRanges[Species].Start * NFrames * 3;
                    Array.Copy(SpeciesResult, 0, Result, Offset, SpeciesResult.Length);
                }

                return Result;
            };

            Func<double[]> GetPerFrameCC = () =>
            {
                double[] Result = new double[NFrames * 3];
                float[] RawResult = GetRawCC(new float2(0));

                for (int p = 0; p < NParticlesOverall; p++)
                for (int f = 0; f < RedNFrames; f++)
                {
                    Result[f * 3 + 0] += RawResult[(p * NFrames + f) * 3 + 0];
                    Result[f * 3 + 1] += RawResult[(p * NFrames + f) * 3 + 1];
                    Result[f * 3 + 2] += RawResult[(p * NFrames + f) * 3 + 2];
                }

                Result = Helper.ArrayOfFunction(f => Result[f * 3 + 0] /
                                                     Math.Max(1e-10, Math.Sqrt(Result[f * 3 + 1] * Result[f * 3 + 2])) *
                                                     100 * NParticlesOverall,
                    NFrames);

                return Result;
            };

            Func<double[]> GetPerParticleCC = () =>
            {
                double[] Result = new double[NParticlesOverall * 3];
                float[] RawResult = GetRawCC(new float2(0));

                for (int p = 0; p < NParticlesOverall; p++)
                for (int f = 0; f < RedNFrames; f++)
                {
                    Result[p * 3 + 0] += RawResult[(p * NFrames + f) * 3 + 0];
                    Result[p * 3 + 1] += RawResult[(p * NFrames + f) * 3 + 1];
                    Result[p * 3 + 2] += RawResult[(p * NFrames + f) * 3 + 2];
                }

                Result = Helper.ArrayOfFunction(p => Result[p * 3 + 0] /
                                                     Math.Max(1e-10, Math.Sqrt(Result[p * 3 + 1] * Result[p * 3 + 2])) *
                                                     100 * NFrames, NParticlesOverall);

                return Result;
            };

            Func<Species, double[]> GetPerParticleCCSpecies = (species) =>
            {
                Particle[] Particles = SpeciesParticles[species];
                int NParticles = Particles.Length;

                double[] Result = new double[NParticles * 3];
                float[] RawResult = GetRawCC(new float2(0));

                for (int p = 0; p < NParticles; p++)
                for (int f = 0; f < RedNFrames; f++)
                {
                    Result[p * 3 + 0] += RawResult[(p * NFrames + f) * 3 + 0];
                    Result[p * 3 + 1] += RawResult[(p * NFrames + f) * 3 + 1];
                    Result[p * 3 + 2] += RawResult[(p * NFrames + f) * 3 + 2];
                }

                Result = Helper.ArrayOfFunction(p => Result[p * 3 + 0] /
                                                     Math.Max(1e-10, Math.Sqrt(Result[p * 3 + 1] * Result[p * 3 + 2])) *
                                                     100 * NFrames, NParticles);

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
                //{
                //    Result[i] = new float2((h_ScoresXP[i] - h_ScoresXM[i]) / Delta2 * 100,
                //                           (h_ScoresYP[i] - h_ScoresYM[i]) / Delta2 * 100);
                //    //if (float.IsNaN(Result[i].X) || float.IsNaN(Result[i].Y))
                //    //    throw new Exception();
                //}

                return (h_ScoresXP, h_ScoresXM, h_ScoresYP, h_ScoresYM, Delta2);
            };

            #endregion

            #region BFGS prerequisites

            float2[][] OriginalOffsets = Helper.ArrayOfFunction(p => Helper.ArrayOfFunction(t =>
                        new float2(PyramidShiftX[p].Values[t],
                            PyramidShiftY[p].Values[t]),
                    PyramidShiftX[p].Values.Length),
                PyramidShiftX.Count);

            float[] OriginalAngleX = GridAngleX.Values.ToArray();
            float[] OriginalAngleY = GridAngleY.Values.ToArray();
            float[] OriginalAngleZ = GridAngleZ.Values.ToArray();

            float4[] ParticleMagChanges = new float4[ParticleMags.Length];

            float[] OriginalParamsCTF =
            {
                (float)CTF.PhaseShift,
                CTF.Distortion.M11,
                CTF.Distortion.M21,
                CTF.Distortion.M12,
                CTF.Distortion.M22
            };

            CTFOptimizationTypes[] CTFStepTypes =
            {
                CTFOptimizationTypes.Defocus,
                CTFOptimizationTypes.AstigmatismDelta,
                CTFOptimizationTypes.AstigmatismAngle,
                CTFOptimizationTypes.Cs,
                CTFOptimizationTypes.Doming,
                CTFOptimizationTypes.Zernike2,
                CTFOptimizationTypes.Zernike2,
                CTFOptimizationTypes.Zernike2,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.Zernike4,
                CTFOptimizationTypes.PhaseShift,
                CTFOptimizationTypes.Distortion,
                CTFOptimizationTypes.Distortion,
                CTFOptimizationTypes.Distortion,
                CTFOptimizationTypes.Distortion
            };

            float[] OriginalDefocusDelta = GridCTFDefocusDelta.Values.ToList().ToArray();
            float[] OriginalDefocusAngle = GridCTFDefocusAngle.Values.ToList().ToArray();
            float[] OriginalCs = GridCTFCs.Values.ToList().ToArray();

            float[] OriginalDefocusDoming = GridCTFDoming.Values.ToList().ToArray();

            float[] OriginalZernikeOdd = CTF.ZernikeCoeffsOdd.ToList().ToArray();
            float[] OriginalZernikeEven = CTF.ZernikeCoeffsEven.ToList().ToArray();

            Matrix2 OriginalMagnification = MagnificationCorrection.GetCopy();

            float3[][] OriginalParticlePositions = allSpecies.Select(s => Helper.Combine(SpeciesParticles[s].Select(p => p.Coordinates))).ToArray();
            float3[][] OriginalParticleAngles = allSpecies.Select(s => Helper.Combine(SpeciesParticles[s].Select(p => p.Angles))).ToArray();

            int BFGSIterations = 0;
            WarpOptimizationTypes CurrentOptimizationTypeWarp = 0;
            CTFOptimizationTypes CurrentOptimizationTypeCTF = 0;

            double[] InitialParametersWarp = new double[PyramidShiftX.Select(g => g.Values.Length).Sum() * 2 +
                                                        GridAngleX.Values.Length * 3 +
                                                        OriginalParticlePositions.Select(a => a.Length).Sum() * 2 +
                                                        OriginalParticleAngles.Select(a => a.Length).Sum() * 3 +
                                                        CTF.ZernikeCoeffsOdd.Length +
                                                        9 * 4 +
                                                        4];
            double[] InitialParametersDefocus = new double[NParticlesOverall +
                                                           GridCTFDefocusDelta.Values.Length +
                                                           GridCTFDefocusAngle.Values.Length +
                                                           GridCTFCs.Values.Length +
                                                           GridCTFDoming.Values.Length +
                                                           CTF.ZernikeCoeffsEven.Length +
                                                           OriginalParamsCTF.Length];

            #endregion

            #region Set parameters from vector

            Action<double[], Movie, bool> SetWarpFromVector = (input, movie, setParticles) =>
            {
                int Offset = 0;

                int3[] PyramidDimensions = PyramidShiftX.Select(g => g.Dimensions).ToArray();

                movie.PyramidShiftX.Clear();
                movie.PyramidShiftY.Clear();

                for (int p = 0; p < PyramidDimensions.Length; p++)
                {
                    float[] MovementXData = new float[PyramidDimensions[p].Elements()];
                    float[] MovementYData = new float[PyramidDimensions[p].Elements()];
                    for (int i = 0; i < MovementXData.Length; i++)
                    {
                        MovementXData[i] = OriginalOffsets[p][i].X + (float)input[Offset + i * 2 + 0];
                        MovementYData[i] = OriginalOffsets[p][i].Y + (float)input[Offset + i * 2 + 1];
                    }

                    movie.PyramidShiftX.Add(new CubicGrid(PyramidDimensions[p], MovementXData));
                    movie.PyramidShiftY.Add(new CubicGrid(PyramidDimensions[p], MovementYData));

                    Offset += MovementXData.Length * 2;
                }

                float[] AngleXData = new float[GridAngleX.Values.Length];
                float[] AngleYData = new float[GridAngleY.Values.Length];
                float[] AngleZData = new float[GridAngleZ.Values.Length];
                for (int i = 0; i < AngleXData.Length; i++)
                {
                    AngleXData[i] = OriginalAngleX[i] + (float)input[Offset + i];
                    AngleYData[i] = OriginalAngleY[i] + (float)input[Offset + AngleXData.Length + i];
                    AngleZData[i] = OriginalAngleZ[i] + (float)input[Offset + AngleXData.Length * 2 + i];
                }

                movie.GridAngleX = new CubicGrid(GridAngleX.Dimensions, AngleXData);
                movie.GridAngleY = new CubicGrid(GridAngleY.Dimensions, AngleYData);
                movie.GridAngleZ = new CubicGrid(GridAngleZ.Dimensions, AngleZData);

                Offset += AngleXData.Length * 3;

                if (setParticles)
                {
                    for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                    {
                        Particle[] Particles = SpeciesParticles[allSpecies[ispecies]];

                        int ResCoords = allSpecies[0].TemporalResolutionMovement;

                        for (int p = 0; p < Particles.Length; p++)
                        {
                            for (int ic = 0; ic < ResCoords; ic++)
                            {
                                Particles[p].Coordinates[ic].X = OriginalParticlePositions[ispecies][p * ResCoords + ic].X + (float)input[Offset + (p * 5 + 0) * ResCoords + ic];
                                Particles[p].Coordinates[ic].Y = OriginalParticlePositions[ispecies][p * ResCoords + ic].Y + (float)input[Offset + (p * 5 + 1) * ResCoords + ic];

                                Particles[p].Angles[ic] = OriginalParticleAngles[ispecies][p * ResCoords + ic] + new float3((float)input[Offset + (p * 5 + 2) * ResCoords + ic],
                                    (float)input[Offset + (p * 5 + 3) * ResCoords + ic],
                                    (float)input[Offset + (p * 5 + 4) * ResCoords + ic]);
                            }
                        }

                        Offset += OriginalParticlePositions[ispecies].Length * 5;
                    }
                }
                else
                {
                    Offset += OriginalParticlePositions.Select(a => a.Length).Sum() * 5;
                }

                for (int icoeff = 0; icoeff < CTF.ZernikeCoeffsOdd.Length; icoeff++)
                    CTF.ZernikeCoeffsOdd[icoeff] = OriginalZernikeOdd[icoeff] + (float)input[Offset + icoeff];

                // In beam tilt, coeff 0 is coupled with 3, and 1 with 4
                if (CTF.ZernikeCoeffsOdd.Length >= 5)
                {
                    CTF.ZernikeCoeffsOdd[0] = 2 * CTF.ZernikeCoeffsOdd[3];
                    CTF.ZernikeCoeffsOdd[1] = 2 * CTF.ZernikeCoeffsOdd[4];
                }

                Offset += CTF.ZernikeCoeffsOdd.Length;

                {
                    float4 MagCorrGlobal = MagnificationCorrection.ToVec();

                    CubicGrid Grid11 = new CubicGrid(new int3(3, 3, 1), input.Skip(Offset + 0).Take(9).Select(v => (float)v).ToArray());
                    CubicGrid Grid21 = new CubicGrid(new int3(3, 3, 1), input.Skip(Offset + 9).Take(9).Select(v => (float)v).ToArray());
                    CubicGrid Grid12 = new CubicGrid(new int3(3, 3, 1), input.Skip(Offset + 18).Take(9).Select(v => (float)v).ToArray());
                    CubicGrid Grid22 = new CubicGrid(new int3(3, 3, 1), input.Skip(Offset + 27).Take(9).Select(v => (float)v).ToArray());

                    float3[] Coords = SpeciesParticles[allSpecies[0]].Select(p => new float3(p.Coordinates[0].X / ImageDimensionsPhysical.X, p.Coordinates[0].Y / ImageDimensionsPhysical.Y, 0.5f)).ToArray();

                    float[] M11 = Grid11.GetInterpolated(Coords);
                    float[] M21 = Grid21.GetInterpolated(Coords);
                    float[] M12 = Grid12.GetInterpolated(Coords);
                    float[] M22 = Grid22.GetInterpolated(Coords);

                    for (int p = 0; p < ParticleMags.Length; p++)
                    {
                        ParticleMags[p] = MagCorrGlobal + new float4(M11[p], M21[p], M12[p], M22[p]) * 0.01f;
                    }

                    Offset += 9 * 4;
                }

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

            Action<double[], Movie, bool> SetDefocusFromVector = (input, movie, setParticles) =>
            {
                int Offset = 0;

                if (setParticles)
                {
                    for (int ispecies = 0; ispecies < allSpecies.Length; ispecies++)
                    {
                        Particle[] Particles = SpeciesParticles[allSpecies[ispecies]];

                        int ResCoords = allSpecies[ispecies].TemporalResolutionMovement;

                        for (int p = 0; p < Particles.Length; p++)
                        {
                            // Coords are in Angstrom and we want 0.1 * micrometer, thus * 1e3
                            for (int ic = 0; ic < ResCoords; ic++)
                                Particles[p].Coordinates[ic].Z = OriginalParticlePositions[ispecies][p * ResCoords + ic].Z + (float)input[p + Offset] * 1e3f;
                        }

                        Offset += Particles.Length;
                    }
                }
                else
                {
                    Offset += NParticlesOverall;
                }

                {
                    float[] ValuesDelta = new float[GridCTFDefocusDelta.Values.Length];
                    for (int i = 0; i < ValuesDelta.Length; i++)
                        ValuesDelta[i] = OriginalDefocusDelta[i] + (float)input[Offset + i] * 0.1f;

                    movie.GridCTFDefocusDelta = new CubicGrid(GridCTFDefocusDelta.Dimensions, ValuesDelta);
                    Offset += ValuesDelta.Length;
                }

                {
                    float[] ValuesAngle = new float[GridCTFDefocusAngle.Values.Length];
                    for (int i = 0; i < ValuesAngle.Length; i++)
                        ValuesAngle[i] = OriginalDefocusAngle[i] + (float)input[Offset + i] * 36;

                    movie.GridCTFDefocusAngle = new CubicGrid(GridCTFDefocusAngle.Dimensions, ValuesAngle);
                    Offset += ValuesAngle.Length;
                }

                {
                    float[] ValuesCs = new float[GridCTFCs.Values.Length];
                    for (int i = 0; i < ValuesCs.Length; i++)
                        ValuesCs[i] = OriginalCs[i] + (float)input[Offset + i] * 3;

                    movie.GridCTFCs = new CubicGrid(GridCTFCs.Dimensions, ValuesCs);
                    Offset += ValuesCs.Length;
                }

                {
                    float[] ValuesDoming = new float[GridCTFDoming.Values.Length];
                    for (int i = 0; i < ValuesDoming.Length; i++)
                        ValuesDoming[i] = OriginalDefocusDoming[i] + (float)input[Offset + i] * 0.1f;

                    movie.GridCTFDoming = new CubicGrid(GridCTFDoming.Dimensions, ValuesDoming);
                    Offset += ValuesDoming.Length;
                }

                {
                    float[] ValuesZernike = new float[CTF.ZernikeCoeffsEven.Length];
                    for (int i = 0; i < ValuesZernike.Length; i++)
                        ValuesZernike[i] = OriginalZernikeEven[i] + (float)input[Offset + i];

                    movie.CTF.ZernikeCoeffsEven = ValuesZernike;
                    Offset += CTF.ZernikeCoeffsEven.Length;
                }

                movie.CTF.PhaseShift = (decimal)(OriginalParamsCTF[0] + input[input.Length - 5]);

                movie.CTF.Distortion = new Matrix2(OriginalParamsCTF[1] + (float)input[input.Length - 4] / 100,
                    OriginalParamsCTF[2] + (float)input[input.Length - 3] / 100,
                    OriginalParamsCTF[3] + (float)input[input.Length - 2] / 100,
                    OriginalParamsCTF[4] + (float)input[input.Length - 1] / 100);
                movie.CTF.PixelSizeDeltaPercent = 0;
            };

            #endregion

            #region Wiggle weights

            progressCallback("Precomputing gradient weights...");

            int NWiggleDifferentiableWarp = PyramidShiftX.Select(g => g.Values.Length).Sum() * 2;
            (int[] indices, float2[] weights)[] AllWiggleWeightsWarp = new (int[] indices, float2[] weights)[NWiggleDifferentiableWarp];

            int NWiggleDifferentiableAstigmatism = GridCTFDefocusDelta.Values.Length;
            (int[] indices, float[] weights)[] AllWiggleWeightsAstigmatism = new (int[] indices, float[] weights)[NWiggleDifferentiableAstigmatism];

            int NWiggleDifferentiableDoming = GridCTFDoming.Values.Length;
            (int[] indices, float[] weights)[] AllWiggleWeightsDoming = new (int[] indices, float[] weights)[NWiggleDifferentiableDoming];

            //if ((optionsMPA.RefinedComponentsWarp & WarpOptimizationTypes.ImageWarp) != 0 ||
            //    (optionsMPA.RefinedComponentsCTF & CTFOptimizationTypes.AstigmatismDelta) != 0)
            {
                Movie[] ParallelMovieCopies = Helper.ArrayOfFunction(i => new Movie(this.Path), 32);

                Dictionary<Species, float3[]> SpeciesParticlePositions = new Dictionary<Species, float3[]>();
                Dictionary<Species, float2[]> SpeciesParticleAstigmatism = new Dictionary<Species, float2[]>();
                foreach (var species in allSpecies)
                {
                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    float3[] ParticlePositions = new float3[NParticles * NFrames];
                    for (int p = 0; p < NParticles; p++)
                    {
                        float3[] Positions = Particles[p].GetCoordinateSeries(DoseInterpolationSteps);

                        for (int f = 0; f < NFrames; f++)
                            ParticlePositions[p * NFrames + f] = Positions[f];
                    }

                    SpeciesParticlePositions.Add(species, ParticlePositions);

                    SpeciesParticleAstigmatism.Add(species, GetAstigmatism(Particles.Select(p => p.Coordinates[0]).ToArray()));
                }

                #region Warp

                if (optionsMPA.DoImageWarp || optionsMPA.DoZernike13 || optionsMPA.DoZernike5)
                    Helper.ForCPU(0, NWiggleDifferentiableWarp / 2, ParallelMovieCopies.Length, (threadID) =>
                        {
                            ParallelMovieCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                            ParallelMovieCopies[threadID].NFrames = NFrames;
                            ParallelMovieCopies[threadID].FractionFrames = FractionFrames;
                        },
                        (iwiggle, threadID) =>
                        {
                            double[] WiggleParams = new double[InitialParametersWarp.Length];
                            WiggleParams[iwiggle * 2] = 1;
                            SetWarpFromVector(WiggleParams, ParallelMovieCopies[threadID], false);

                            float2[] RawShifts = new float2[NParticlesOverall * NFrames];
                            foreach (var species in allSpecies)
                            {
                                Particle[] Particles = SpeciesParticles[species];
                                int NParticles = Particles.Length;
                                if (NParticles == 0)
                                    continue;

                                int Offset = SpeciesParticleIDRanges[species].Start;

                                float3[] ParticlePositionsProjected = ParallelMovieCopies[threadID].GetPositionInAllFrames(SpeciesParticlePositions[species]);
                                float2[] ParticleExtractedAt = SpeciesParticleExtractedAt[species];

                                for (int p = 0; p < NParticles; p++)
                                for (int f = 0; f < NFrames; f++)
                                    RawShifts[(Offset + p) * NFrames + f] = (new float2(ParticlePositionsProjected[p * NFrames + f]) - ParticleExtractedAt[p * NFrames + f]);
                            }

                            List<int> Indices = new List<int>(RawShifts.Length / 5);
                            List<float2> Weights = new List<float2>(RawShifts.Length / 5);
                            List<float2> WeightsY = new List<float2>(RawShifts.Length / 5);
                            for (int i = 0; i < RawShifts.Length; i++)
                            {
                                if (RawShifts[i].LengthSq() > 1e-6f)
                                {
                                    Indices.Add(i);
                                    Weights.Add(RawShifts[i]);
                                    WeightsY.Add(new float2(RawShifts[i].Y, RawShifts[i].X));

                                    if (Math.Abs(RawShifts[i].X) > 1.5f)
                                        throw new Exception();
                                }
                            }

                            AllWiggleWeightsWarp[iwiggle * 2 + 0] = (Indices.ToArray(), Weights.ToArray());
                            AllWiggleWeightsWarp[iwiggle * 2 + 1] = (Indices.ToArray(), WeightsY.ToArray());
                        }, null);

                #endregion

                #region Astigmatism

                if (optionsMPA.DoAstigmatismDelta || optionsMPA.DoCs)
                    Helper.ForCPU(0, NWiggleDifferentiableAstigmatism, ParallelMovieCopies.Length, (threadID) =>
                        {
                            ParallelMovieCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                            ParallelMovieCopies[threadID].NFrames = NFrames;
                            ParallelMovieCopies[threadID].FractionFrames = FractionFrames;
                        },
                        (iwiggle, threadID) =>
                        {
                            double[] WiggleParams = new double[InitialParametersDefocus.Length];
                            WiggleParams[NParticlesOverall + iwiggle] = 10; // because it's weighted *0.1 later in SetDefocusFromVector
                            SetDefocusFromVector(WiggleParams, ParallelMovieCopies[threadID], false);

                            float2[] RawDiffs = new float2[NParticlesOverall];
                            foreach (var species in allSpecies)
                            {
                                Particle[] Particles = SpeciesParticles[species];
                                int NParticles = Particles.Length;
                                if (NParticles == 0)
                                    continue;

                                int Offset = SpeciesParticleIDRanges[species].Start;

                                float2[] ParticleAstigmatismAltered = ParallelMovieCopies[threadID].GetAstigmatism(Particles.Select(p => p.Coordinates[0]).ToArray());
                                float2[] ParticleAstigmatismOriginal = SpeciesParticleAstigmatism[species];

                                for (int p = 0; p < NParticles; p++)
                                    RawDiffs[Offset + p] = ParticleAstigmatismAltered[p] - ParticleAstigmatismOriginal[p];
                            }

                            List<int> Indices = new List<int>(RawDiffs.Length);
                            List<float> Weights = new List<float>(RawDiffs.Length);
                            for (int i = 0; i < RawDiffs.Length; i++)
                            {
                                Indices.Add(i);
                                Weights.Add(RawDiffs[i].X);
                            }

                            AllWiggleWeightsAstigmatism[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                        }, null);

                #endregion

                #region Doming

                //if ((optionsMPA.RefinedComponentsCTF & CTFOptimizationTypes.Doming) != 0)
                Helper.ForCPU(0, NWiggleDifferentiableDoming, ParallelMovieCopies.Length, (threadID) =>
                    {
                        ParallelMovieCopies[threadID].ImageDimensionsPhysical = ImageDimensionsPhysical;
                        ParallelMovieCopies[threadID].NFrames = NFrames;
                        ParallelMovieCopies[threadID].FractionFrames = FractionFrames;
                    },
                    (iwiggle, threadID) =>
                    {
                        double[] WiggleParams = new double[InitialParametersDefocus.Length];
                        WiggleParams[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + iwiggle] = 10; // because it's weighted *0.1 later in SetDefocusFromVector
                        SetDefocusFromVector(WiggleParams, ParallelMovieCopies[threadID], false);

                        float[] RawDefoci = new float[NParticlesOverall * NFrames];
                        foreach (var species in allSpecies)
                        {
                            Particle[] Particles = SpeciesParticles[species];
                            int NParticles = Particles.Length;
                            if (NParticles == 0)
                                continue;

                            int Offset = SpeciesParticleIDRanges[species].Start;

                            float3[] ParticlePositionsProjected = ParallelMovieCopies[threadID].GetPositionInAllFrames(SpeciesParticlePositions[species]);
                            float[] ParticleExtractedAt = SpeciesParticleExtractedAtDefocus[species];

                            for (int p = 0; p < NParticles; p++)
                            for (int f = 0; f < NFrames; f++)
                                RawDefoci[(Offset + p) * NFrames + f] = ParticlePositionsProjected[p * NFrames + f].Z - ParticleExtractedAt[p * NFrames + f];
                        }

                        List<int> Indices = new List<int>(RawDefoci.Length / NFrames);
                        List<float> Weights = new List<float>(RawDefoci.Length / NFrames);
                        for (int i = 0; i < RawDefoci.Length; i++)
                        {
                            if (Math.Abs(RawDefoci[i]) > 1e-6f)
                            {
                                Indices.Add(i);
                                Weights.Add(RawDefoci[i]);

                                if (Math.Abs(RawDefoci[i]) > 1.5f)
                                    throw new Exception();
                            }
                        }

                        AllWiggleWeightsDoming[iwiggle] = (Indices.ToArray(), Weights.ToArray());
                    }, null);

                #endregion
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
                    for (int f = 0; f < RedNFrames; f++)
                    {
                        SumAB += RawCC[(p * NFrames + f) * 3 + 0];
                        SumA2 += RawCC[(p * NFrames + f) * 3 + 1];
                        SumB2 += RawCC[(p * NFrames + f) * 3 + 2];
                    }
                }

                double Score = SumAB / Math.Max(1e-10, Math.Sqrt(SumA2 * SumB2)) * NParticlesOverall * NFrames * 100;

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

                if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ImageWarp) != 0) // Image shift pyramids
                {
                    SetWarpFromVector(input, this, true);
                    (var XP, var XM, var YP, var YM, var Delta2Movement) = GetRawShiftGradients();

                    Parallel.For(0, AllWiggleWeightsWarp.Length, iwiggle =>
                    {
                        double SumGrad = 0;
                        double SumWeights = 0;
                        double SumWeightsGrad = 0;

                        int[] Indices = AllWiggleWeightsWarp[iwiggle].indices;
                        float2[] Weights = AllWiggleWeightsWarp[iwiggle].weights;

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
                }

                Offset += AllWiggleWeightsWarp.Length;


                if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.AxisAngle) != 0) // GridAngleX, Y and Z
                {
                    int SliceElements = (int)GridAngleX.Dimensions.ElementsSlice();

                    for (int a = 0; a < 3; a++)
                    {
                        for (int i = 0; i < SliceElements; i++)
                        {
                            double[] InputPlus = input.ToArray();
                            for (int t = 0; t < NFrames; t++)
                                InputPlus[Offset + t * SliceElements + i] += Delta;

                            SetWarpFromVector(InputPlus, this, true);
                            double[] ScoresPlus = GetPerFrameCC();

                            double[] InputMinus = input.ToArray();
                            for (int t = 0; t < NFrames; t++)
                                InputMinus[Offset + t * SliceElements + i] -= Delta;

                            SetWarpFromVector(InputMinus, this, true);
                            double[] ScoresMinus = GetPerFrameCC();

                            for (int t = 0; t < NFrames; t++)
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
                            for (int iparam = 0; iparam < 2 * TemporalRes; iparam++)
                            {
                                if (iparam % TemporalRes != 0)
                                    continue;

                                double[] InputPlus = input.ToArray();
                                for (int p = 0; p < Particles.Length; p++)
                                    InputPlus[Offset + p * 5 * TemporalRes + iparam] += Delta;

                                SetWarpFromVector(InputPlus, this, true);
                                double[] ScoresPlus = GetPerParticleCCSpecies(Species);

                                double[] InputMinus = input.ToArray();
                                for (int p = 0; p < Particles.Length; p++)
                                    InputMinus[Offset + p * 5 * TemporalRes + iparam] -= Delta;

                                SetWarpFromVector(InputMinus, this, true);
                                double[] ScoresMinus = GetPerParticleCCSpecies(Species);

                                for (int p = 0; p < Particles.Length; p++)
                                    Result[Offset + p * 5 * TemporalRes + iparam] = (ScoresPlus[p] - ScoresMinus[p]) / Delta2;
                            }

                        if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ParticleAngle) != 0)
                            for (int iparam = 2 * TemporalRes; iparam < 5 * TemporalRes; iparam++)
                            {
                                if (iparam % TemporalRes != 0)
                                    continue;

                                double[] InputPlus = input.ToArray();
                                for (int p = 0; p < Particles.Length; p++)
                                    InputPlus[Offset + p * 5 * TemporalRes + iparam] += Delta;

                                SetWarpFromVector(InputPlus, this, true);
                                double[] ScoresPlus = GetPerParticleCCSpecies(Species);

                                double[] InputMinus = input.ToArray();
                                for (int p = 0; p < Particles.Length; p++)
                                    InputMinus[Offset + p * 5 * TemporalRes + iparam] -= Delta;

                                SetWarpFromVector(InputMinus, this, true);
                                double[] ScoresMinus = GetPerParticleCCSpecies(Species);

                                for (int p = 0; p < Particles.Length; p++)
                                    Result[Offset + p * 5 * TemporalRes + iparam] = (ScoresPlus[p] - ScoresMinus[p]) / Delta2;
                            }

                        Offset += OriginalParticlePositions[ispecies].Length * 5; // No * TemporalRes because it's already included in OriginalParticlePositions
                    }
                }

                if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.Zernike13) != 0)
                {
                    // First 2 coeffs are coupled with 3 and 4 for beam tilt
                    for (int iparam = 2; iparam < Math.Min(6, CTF.ZernikeCoeffsOdd.Length); iparam++)
                    {
                        double[] InputPlus = input.ToArray();
                        InputPlus[Offset + iparam] += Delta;

                        double ScoresPlus = WarpEval(InputPlus);

                        double[] InputMinus = input.ToArray();
                        InputMinus[Offset + iparam] -= Delta;

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

                if ((CurrentOptimizationTypeWarp & WarpOptimizationTypes.ParticleMag) != 0)
                    for (int iparam = 0; iparam < 9 * 4; iparam++)
                    {
                        double[] InputPlus = input.ToArray();
                        InputPlus[Offset + iparam] += Delta;
                        double ScoresPlus = WarpEval(InputPlus);

                        double[] InputMinus = input.ToArray();
                        InputMinus[Offset + iparam] -= Delta;
                        double ScoresMinus = WarpEval(InputMinus);

                        Result[Offset + iparam] = (ScoresPlus - ScoresMinus) / Delta2;
                    }

                Offset += 9 * 4;

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

            Func<double[], double> CTFEval = input =>
            {
                SetDefocusFromVector(input, this, true);

                double ScoreAB = 0, ScoreA2 = 0, ScoreB2 = 0;

                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                foreach (var species in allSpecies)
                {
                    if (!SpeciesParticles.ContainsKey(species))
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    int SuperresFactor = SpeciesCTFSuperresFactor[species];
                    int BatchSize = optionsMPA.BatchSize;
                    BatchSize /= SuperresFactor * SuperresFactor;

                    float[] ResultP = new float[BatchSize * 3];
                    float[] ResultQ = new float[BatchSize * 3];

                    int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeRefineSuper = SizeRefine * SuperresFactor;
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * SuperresFactor;

                    float AngPixRefine = species.ResolutionRefinement / 2;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                    Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion); // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine

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
                    Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                    for (int f = 0; f < NFrames; f++)
                        GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                            PhaseCorrectionAll.GetDeviceSlice(f, Intent.Write),
                            PhaseCorrection.Dims.Slice(),
                            new int3(RelevantSizes[f]).Slice(),
                            1);
                    PhaseCorrection.Dispose();

                    Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                    bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                    float[][] EwaldResults = { ResultP, ResultQ };

                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                        IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                        float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                        float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                        for (int f = 0; f < RedNFrames; f++)
                        {
                            float3[] CoordinatesFrame = new float3[CurBatch];
                            float3[] AnglesFrame = new float3[CurBatch];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                AnglesFrame[p] = AnglesMoving[p * NFrames + f]; // * Helper.ToRad;
                            }

                            float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                            float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f); // AnglesFrame;
                            float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);

                            float3[] Defoci = new float3[CurBatch];
                            int3[] ExtractOrigins = new int3[CurBatch];
                            float3[] ResidualShifts = new float3[BatchSize];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                Defoci[p].X = ImageCoords[p].Z;
                                Defoci[p].Y = Astigmatism[p].X;
                                Defoci[p].Z = Astigmatism[p].Y;
                            }

                            for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                            {
                                GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    FrameData[f].Dims.Slice(),
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
                                    GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, EwaldReverse[iewald], ExtractedCTF, true);

                                    GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                        ExtractedCTF.GetDevice(Intent.Read),
                                        ExtractedFT.GetDevice(Intent.Write),
                                        ExtractedCTF.ElementsComplex,
                                        1);
                                }
                                else
                                {
                                    GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                    ParticleDiameterPix / 2f * 1.3f,
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
                                    ExtractedCropped.GetDevice(Intent.Write),
                                    new int3(SizeRefine).Slice(),
                                    new int3(RelevantSizes[f]).Slice(),
                                    (uint)CurBatch);


                                GPU.MultiParticleDiff(EwaldResults[iewald],
                                    new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                    SizeRefine,
                                    new[] { RelevantSizes[f] },
                                    new float[CurBatch * 2],
                                    Helper.ToInterleaved(ImageAngles),
                                    MagnificationCorrection.ToVec(),
                                    SpeciesCTFWeights[species].GetDeviceSlice(f, Intent.Read),
                                    PhaseCorrectionAll.GetDeviceSlice(f, Intent.Read),
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
                                ScoreAB += ResultP[i * 3 + 0] + ResultQ[i * 3 + 0];
                                ScoreA2 += ResultP[i * 3 + 1] + ResultQ[i * 3 + 1];
                                ScoreB2 += ResultP[i * 3 + 2] + ResultQ[i * 3 + 2];
                                //Debug.WriteLine(Result[i * 3 + 0]);
                                //Debug.WriteLine(Result[i * 3 + 1]);
                                //Debug.WriteLine(Result[i * 3 + 2]);
                            }
                        }
                    }

                    PhaseCorrectionAll.Dispose();
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

                //foreach (var image in FrameData)
                //    image.FreeDevice();

                double Score = ScoreAB / Math.Max(1e-10, Math.Sqrt(ScoreA2 * ScoreB2)) * NParticlesOverall * NFrames;
                Score *= 100;

                Console.WriteLine(Score);

                return Score;
            };

            Func<double[], double[]> CTFGrad = input =>
            {
                //Stopwatch Watch = new Stopwatch();
                //Watch.Start();

                double Delta = 0.001;
                double Delta2 = Delta * 2;

                double[] Deltas = { Delta, -Delta };

                double[] Result = new double[input.Length];
                double[] ScoresAB = new double[input.Length * 2];
                double[] ScoresA2 = new double[input.Length * 2];
                double[] ScoresB2 = new double[input.Length * 2];
                int[] ScoresSamples = new int[input.Length * 2];

                float[][] PerParticleScoresAB = Helper.ArrayOfFunction(i => new float[(CTFStepTypes[i] & CTFOptimizationTypes.Doming) == 0 ? NParticlesOverall * 2 : NParticlesOverall * NFrames * 2], CTFStepTypes.Length);
                float[][] PerParticleScoresA2 = Helper.ArrayOfFunction(i => new float[(CTFStepTypes[i] & CTFOptimizationTypes.Doming) == 0 ? NParticlesOverall * 2 : NParticlesOverall * NFrames * 2], CTFStepTypes.Length);
                float[][] PerParticleScoresB2 = Helper.ArrayOfFunction(i => new float[(CTFStepTypes[i] & CTFOptimizationTypes.Doming) == 0 ? NParticlesOverall * 2 : NParticlesOverall * NFrames * 2], CTFStepTypes.Length);

                if (BFGSIterations-- <= 0)
                    return Result;

                if (MathHelper.AllEqual(input, OldInput))
                    return OldGradient;

                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                foreach (var species in allSpecies)
                {
                    if (!SpeciesParticles.ContainsKey(species) ||
                        species.ResolutionRefinement > (float)optionsMPA.MinimumCTFRefinementResolution)
                        continue;

                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    int SuperresFactor = SpeciesCTFSuperresFactor[species];
                    int BatchSize = optionsMPA.BatchSize;
                    BatchSize /= SuperresFactor * SuperresFactor;

                    float[] ResultP = new float[BatchSize * 3];
                    float[] ResultQ = new float[BatchSize * 3];

                    int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeRefineSuper = SizeRefine * SuperresFactor;
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * SuperresFactor;

                    int SpeciesOffset = SpeciesParticleIDRanges[species].Start;

                    float AngPixRefine = species.ResolutionRefinement / 2;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                    Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);

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
                    Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                    for (int t = 0; t < NFrames; t++)
                        GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                            PhaseCorrectionAll.GetDeviceSlice(t, Intent.Write),
                            PhaseCorrection.Dims.Slice(),
                            new int3(RelevantSizes[t]).Slice(),
                            1);
                    PhaseCorrection.Dispose();

                    Image GammaCorrection = new Image(new int3(SizeRefineSuper, SizeRefineSuper, 1), true);

                    bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                    float[][] EwaldResults = { ResultP, ResultQ };

                    Stopwatch WatchZernike = new Stopwatch();

                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                        IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                        float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                        float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                        for (int f = 0; f < RedNFrames; f++)
                        {
                            float3[] CoordinatesFrame = new float3[CurBatch];
                            float3[] AnglesFrame = new float3[CurBatch];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                AnglesFrame[p] = AnglesMoving[p * NFrames + f]; // * Helper.ToRad;
                            }

                            float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                            float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f); // AnglesFrame;

                            float3[] Defoci = new float3[CurBatch];
                            int3[] ExtractOrigins = new int3[CurBatch];
                            float3[] ResidualShifts = new float3[BatchSize];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                            }

                            GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                FrameData[f].Dims.Slice(),
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


                            SetDefocusFromVector(input, this, true);

                            //WatchZernike.Start();
                            GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper, GammaCorrection);
                            CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion, CoordsCTF);
                            //WatchZernike.Stop();

                            for (int iparam = 0; iparam < CTFStepTypes.Length; iparam++)
                            {
                                if ((CurrentOptimizationTypeCTF & CTFStepTypes[iparam]) == 0)
                                    continue;

                                for (int idelta = 0; idelta < 2; idelta++)
                                {
                                    double[] InputAltered = input.ToArray();

                                    if (CTFStepTypes[iparam] == CTFOptimizationTypes.Defocus)
                                    {
                                        for (int i = 0; i < NParticles; i++)
                                            InputAltered[i] += Deltas[idelta];
                                    }
                                    else if (CTFStepTypes[iparam] == CTFOptimizationTypes.AstigmatismDelta)
                                    {
                                        for (int i = 0; i < GridCTFDefocusDelta.Values.Length; i++)
                                            InputAltered[NParticlesOverall + i] += Deltas[idelta];
                                    }
                                    else if (CTFStepTypes[iparam] == CTFOptimizationTypes.AstigmatismAngle)
                                    {
                                        for (int i = 0; i < GridCTFDefocusAngle.Values.Length; i++)
                                            InputAltered[NParticlesOverall + GridCTFDefocusDelta.Values.Length + i] += Deltas[idelta];
                                    }
                                    else if (CTFStepTypes[iparam] == CTFOptimizationTypes.Cs)
                                    {
                                        for (int i = 0; i < GridCTFCs.Values.Length; i++)
                                            InputAltered[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + i] += Deltas[idelta];
                                    }
                                    else if (CTFStepTypes[iparam] == CTFOptimizationTypes.Doming)
                                    {
                                        for (int i = 0; i < GridCTFDoming.Values.Length; i++)
                                            InputAltered[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + GridCTFCs.Values.Length + i] += Deltas[idelta];
                                    }
                                    else
                                    {
                                        InputAltered[InputAltered.Length - CTFStepTypes.Length + iparam] += Deltas[idelta];
                                    }

                                    SetDefocusFromVector(InputAltered, this, true);

                                    for (int p = 0; p < CurBatch; p++)
                                        CoordinatesFrame[p].Z = Particles[batchStart + p].GetSplineCoordinateZ().Interp(DoseInterpolationSteps[f]);

                                    ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                    float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        Defoci[p].X = ImageCoords[p].Z;
                                        Defoci[p].Y = Astigmatism[p].X;
                                        Defoci[p].Z = Astigmatism[p].Y;
                                    }

                                    //WatchZernike.Start();
                                    if ((CTFStepTypes[iparam] & CTFOptimizationTypes.Zernike2) != 0 || (CTFStepTypes[iparam] & CTFOptimizationTypes.Zernike4) != 0)
                                        GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper, GammaCorrection);
                                    //WatchZernike.Stop();

                                    if ((CTFStepTypes[iparam] & CTFOptimizationTypes.Distortion) != 0)
                                        CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion, CoordsCTF);

                                    for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                    {
                                        if (species.DoEwald)
                                        {
                                            GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, EwaldReverse[iewald], ExtractedCTF, true);

                                            GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                ExtractedCTF.GetDevice(Intent.Read),
                                                ExtractedFT.GetDevice(Intent.Write),
                                                ExtractedCTF.ElementsComplex,
                                                1);
                                        }
                                        else
                                        {
                                            GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                            ParticleDiameterPix / 2f * 1.3f,
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
                                            new int3(RelevantSizes[f]).Slice(),
                                            (uint)CurBatch);

                                        GPU.MultiParticleDiff(EwaldResults[iewald],
                                            new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                            SizeRefine,
                                            new[] { RelevantSizes[f] },
                                            new float[CurBatch * 2],
                                            Helper.ToInterleaved(ImageAngles),
                                            MagnificationCorrection.ToVec(),
                                            SpeciesCTFWeights[species].GetDeviceSlice(f, Intent.Read),
                                            PhaseCorrectionAll.GetDeviceSlice(f, Intent.Read),
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

                                    if ((CTFStepTypes[iparam] & CTFOptimizationTypes.Doming) == 0)
                                    {
                                        for (int i = 0; i < CurBatch; i++)
                                        {
                                            PerParticleScoresAB[iparam][(SpeciesOffset + batchStart + i) * 2 + idelta] += ResultP[i * 3 + 0] + ResultQ[i * 3 + 0];
                                            PerParticleScoresA2[iparam][(SpeciesOffset + batchStart + i) * 2 + idelta] += ResultP[i * 3 + 1] + ResultQ[i * 3 + 1];
                                            PerParticleScoresB2[iparam][(SpeciesOffset + batchStart + i) * 2 + idelta] += ResultP[i * 3 + 2] + ResultQ[i * 3 + 2];
                                        }
                                    }
                                    else
                                    {
                                        for (int i = 0; i < CurBatch; i++)
                                        {
                                            PerParticleScoresAB[iparam][((SpeciesOffset + batchStart + i) * NFrames + f) * 2 + idelta] = ResultP[i * 3 + 0] + ResultQ[i * 3 + 0];
                                            PerParticleScoresA2[iparam][((SpeciesOffset + batchStart + i) * NFrames + f) * 2 + idelta] = ResultP[i * 3 + 1] + ResultQ[i * 3 + 1];
                                            PerParticleScoresB2[iparam][((SpeciesOffset + batchStart + i) * NFrames + f) * 2 + idelta] = ResultP[i * 3 + 2] + ResultQ[i * 3 + 2];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    //Console.WriteLine($"Zernike: {WatchZernike.ElapsedMilliseconds}");

                    CoordsCTF.Dispose();
                    PhaseCorrectionAll.Dispose();
                    Extracted.Dispose();
                    ExtractedFT.Dispose();
                    ExtractedRefineSuper.Dispose();
                    ExtractedCropped.Dispose();
                    ExtractedCroppedFT.Dispose();
                    ExtractedCTF.Dispose();

                    GammaCorrection.Dispose();

                    GPU.DestroyFFTPlan(PlanForwSuper);
                    GPU.DestroyFFTPlan(PlanBackSuper);
                    GPU.DestroyFFTPlan(PlanForw);
                }

                for (int iparam = 0; iparam < CTFStepTypes.Length; iparam++)
                {
                    if ((CurrentOptimizationTypeCTF & CTFStepTypes[iparam]) == 0)
                        continue;

                    if (CTFStepTypes[iparam] == CTFOptimizationTypes.Defocus)
                    {
                        for (int i = 0; i < NParticlesOverall; i++)
                        {
                            double ScorePlus = PerParticleScoresAB[iparam][i * 2 + 0] /
                                               Math.Max(1e-10, Math.Sqrt(PerParticleScoresA2[iparam][i * 2 + 0] *
                                                                         PerParticleScoresB2[iparam][i * 2 + 0]));
                            double ScoreMinus = PerParticleScoresAB[iparam][i * 2 + 1] /
                                                Math.Max(1e-10, Math.Sqrt(PerParticleScoresA2[iparam][i * 2 + 1] *
                                                                          PerParticleScoresB2[iparam][i * 2 + 1]));
                            Result[i] = (ScorePlus - ScoreMinus) / Delta2 * NFrames * 100;
                        }
                    }
                    else if (CTFStepTypes[iparam] == CTFOptimizationTypes.AstigmatismDelta)
                    {
                        for (int iwiggle = 0; iwiggle < NWiggleDifferentiableAstigmatism; iwiggle++)
                        {
                            double SumGrad = 0;
                            double SumWeights = 0;
                            double SumWeightsGrad = 0;

                            int[] Indices = AllWiggleWeightsAstigmatism[iwiggle].indices;
                            float[] Weights = AllWiggleWeightsAstigmatism[iwiggle].weights;

                            for (int i = 0; i < Indices.Length; i++)
                            {
                                int id = Indices[i];
                                float ABPlus = PerParticleScoresAB[iparam][id * 2 + 0];
                                float ABMinus = PerParticleScoresAB[iparam][id * 2 + 1];
                                float A2Plus = PerParticleScoresA2[iparam][id * 2 + 0];
                                float A2Minus = PerParticleScoresA2[iparam][id * 2 + 1];
                                float B2Plus = PerParticleScoresB2[iparam][id * 2 + 0];
                                float B2Minus = PerParticleScoresB2[iparam][id * 2 + 1];

                                SumWeights += Math.Abs(Weights[i]) * Math.Sqrt(A2Plus + A2Minus);
                                SumWeightsGrad += Math.Abs(Weights[i]);

                                double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                               ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                                SumGrad += Weights[i] * Math.Sqrt(A2Plus + A2Minus) * Grad;
                            }

                            Result[NParticlesOverall + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                        }
                    }
                    else if (CTFStepTypes[iparam] == CTFOptimizationTypes.AstigmatismAngle)
                    {
                        for (int iwiggle = 0; iwiggle < NWiggleDifferentiableAstigmatism; iwiggle++)
                        {
                            double SumGrad = 0;
                            double SumWeights = 0;
                            double SumWeightsGrad = 0;

                            int[] Indices = AllWiggleWeightsAstigmatism[iwiggle].indices;
                            float[] Weights = AllWiggleWeightsAstigmatism[iwiggle].weights;

                            for (int i = 0; i < Indices.Length; i++)
                            {
                                int id = Indices[i];
                                float ABPlus = PerParticleScoresAB[iparam][id * 2 + 0];
                                float ABMinus = PerParticleScoresAB[iparam][id * 2 + 1];
                                float A2Plus = PerParticleScoresA2[iparam][id * 2 + 0];
                                float A2Minus = PerParticleScoresA2[iparam][id * 2 + 1];
                                float B2Plus = PerParticleScoresB2[iparam][id * 2 + 0];
                                float B2Minus = PerParticleScoresB2[iparam][id * 2 + 1];

                                SumWeights += Math.Abs(Weights[i]) * Math.Sqrt(A2Plus + A2Minus);
                                SumWeightsGrad += Math.Abs(Weights[i]);

                                double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                               ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                                SumGrad += Weights[i] * Math.Sqrt(A2Plus + A2Minus) * Grad;
                            }

                            Result[NParticlesOverall + GridCTFDefocusDelta.Values.Length + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                        }
                    }
                    else if (CTFStepTypes[iparam] == CTFOptimizationTypes.Cs)
                    {
                        for (int iwiggle = 0; iwiggle < NWiggleDifferentiableAstigmatism; iwiggle++)
                        {
                            double SumGrad = 0;
                            double SumWeights = 0;
                            double SumWeightsGrad = 0;

                            int[] Indices = AllWiggleWeightsAstigmatism[iwiggle].indices;
                            float[] Weights = AllWiggleWeightsAstigmatism[iwiggle].weights;

                            for (int i = 0; i < Indices.Length; i++)
                            {
                                int id = Indices[i];
                                float ABPlus = PerParticleScoresAB[iparam][id * 2 + 0];
                                float ABMinus = PerParticleScoresAB[iparam][id * 2 + 1];
                                float A2Plus = PerParticleScoresA2[iparam][id * 2 + 0];
                                float A2Minus = PerParticleScoresA2[iparam][id * 2 + 1];
                                float B2Plus = PerParticleScoresB2[iparam][id * 2 + 0];
                                float B2Minus = PerParticleScoresB2[iparam][id * 2 + 1];

                                SumWeights += Math.Abs(Weights[i]) * Math.Sqrt(A2Plus + A2Minus);
                                SumWeightsGrad += Math.Abs(Weights[i]);

                                double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                               ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                                SumGrad += Weights[i] * Math.Sqrt(A2Plus + A2Minus) * Grad;
                            }

                            Result[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                        }
                    }
                    else if (CTFStepTypes[iparam] == CTFOptimizationTypes.Doming)
                    {
                        for (int iwiggle = 0; iwiggle < NWiggleDifferentiableDoming; iwiggle++)
                        {
                            double SumGrad = 0;
                            double SumWeights = 0;
                            double SumWeightsGrad = 0;

                            int[] Indices = AllWiggleWeightsDoming[iwiggle].indices;
                            float[] Weights = AllWiggleWeightsDoming[iwiggle].weights;

                            for (int i = 0; i < Indices.Length; i++)
                            {
                                int id = Indices[i];
                                float ABPlus = PerParticleScoresAB[iparam][id * 2 + 0];
                                float ABMinus = PerParticleScoresAB[iparam][id * 2 + 1];
                                float A2Plus = PerParticleScoresA2[iparam][id * 2 + 0];
                                float A2Minus = PerParticleScoresA2[iparam][id * 2 + 1];
                                float B2Plus = PerParticleScoresB2[iparam][id * 2 + 0];
                                float B2Minus = PerParticleScoresB2[iparam][id * 2 + 1];

                                SumWeights += Math.Abs(Weights[i]) * Math.Sqrt(A2Plus + A2Minus);
                                SumWeightsGrad += Math.Abs(Weights[i]);

                                double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                               ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                                SumGrad += Weights[i] * Math.Sqrt(A2Plus + A2Minus) * Grad;
                            }

                            Result[NParticlesOverall + GridCTFDefocusDelta.Values.Length * 2 + GridCTFCs.Values.Length + iwiggle] = SumGrad / Math.Max(1e-15, SumWeights) * 100 * SumWeightsGrad;
                        }
                    }
                    else
                    {
                        double ABPlus = 0, ABMinus = 0;
                        double A2Plus = 0, A2Minus = 0;
                        double B2Plus = 0, B2Minus = 0;
                        for (int i = 0; i < NParticlesOverall; i++)
                        {
                            ABPlus += PerParticleScoresAB[iparam][i * 2 + 0];
                            ABMinus += PerParticleScoresAB[iparam][i * 2 + 1];

                            A2Plus += PerParticleScoresA2[iparam][i * 2 + 0];
                            A2Minus += PerParticleScoresA2[iparam][i * 2 + 1];

                            B2Plus += PerParticleScoresB2[iparam][i * 2 + 0];
                            B2Minus += PerParticleScoresB2[iparam][i * 2 + 1];
                        }

                        double Grad = (ABPlus / Math.Max(1e-15, Math.Sqrt(A2Plus * B2Plus)) -
                                       ABMinus / Math.Max(1e-15, Math.Sqrt(A2Minus * B2Minus))) / Delta2;

                        Result[Result.Length - CTFStepTypes.Length + iparam] = Grad * NParticlesOverall * NFrames * 100;
                    }
                }

                OldInput = input.ToList().ToArray();
                OldGradient = Result.ToList().ToArray();

                //Watch.Stop();
                //Console.WriteLine(Watch.ElapsedMilliseconds);

                return Result;
            };

            #endregion

            #region Grid search for per-particle defoci

            Func<double[], double[]> DefocusGridSearch = input =>
            {
                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;

                foreach (var species in allSpecies)
                {
                    if (!SpeciesParticles.ContainsKey(species))
                        continue;
                    Particle[] Particles = SpeciesParticles[species];
                    int NParticles = Particles.Length;
                    if (NParticles == 0)
                        continue;

                    int SuperresFactor = SpeciesCTFSuperresFactor[species];
                    int BatchSize = optionsMPA.BatchSize;
                    BatchSize /= SuperresFactor * SuperresFactor;

                    float[] ResultP = new float[BatchSize * 3];
                    float[] ResultQ = new float[BatchSize * 3];

                    int SpeciesOffset = SpeciesParticleIDRanges[species].Start;

                    int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
                    int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[species];
                    int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                    int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[species];

                    float AngPixRefine = species.ResolutionRefinement / 2;
                    int ParticleDiameterPix = (int)(species.DiameterAngstrom / AngPixRefine);

                    int[] RelevantSizes = SpeciesRelevantRefinementSizes[species];

                    Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);

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
                    Image PhaseCorrectionAll = new Image(new int3(SizeRefine, SizeRefine, NFrames), true, true);
                    for (int f = 0; f < NFrames; f++)
                        GPU.CropFT(PhaseCorrection.GetDevice(Intent.Read),
                            PhaseCorrectionAll.GetDeviceSlice(f, Intent.Write),
                            PhaseCorrection.Dims.Slice(),
                            new int3(RelevantSizes[f]).Slice(),
                            1);
                    PhaseCorrection.Dispose();

                    Image GammaCorrection = CTF.GetGammaCorrection(AngPixRefine, SizeRefineSuper);

                    bool[] EwaldReverse = { species.EwaldReverse, !species.EwaldReverse };
                    float[][] EwaldResults = { ResultP, ResultQ };

                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                        IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                        float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                        float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                        List<float4>[] AllSearchValues = Helper.ArrayOfFunction(i => new List<float4>(), CurBatch);
                        List<float4>[] CurrentSearchValues = Helper.ArrayOfFunction(i => new List<float4>(), CurBatch);
                        decimal GridSearchDelta = 0.002M;
                        foreach (var list in CurrentSearchValues)
                            for (decimal d = -0.02M; d <= 0.02M; d += GridSearchDelta)
                                list.Add(new float4((float)d, 0, 0, 0));
                        //for (decimal d = 0M; d <= 0M; d += GridSearchDelta)
                        //    list.Add(new float2((float)d, 0));

                        for (int irefine = 0; irefine < 4; irefine++)
                        {
                            for (int f = 0; f < RedNFrames; f++)
                            {
                                float3[] CoordinatesFrame = new float3[CurBatch];
                                float3[] AnglesFrame = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                    AnglesFrame[p] = AnglesMoving[p * NFrames + f]; // * Helper.ToRad;
                                }

                                float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f); // AnglesFrame;

                                float3[] Defoci = new float3[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                }

                                GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                    Extracted.GetDevice(Intent.Write),
                                    FrameData[f].Dims.Slice(),
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

                                for (int idelta = 0; idelta < CurrentSearchValues[0].Count; idelta++)
                                {
                                    double[] InputAltered = input.ToArray();
                                    for (int i = 0; i < CurBatch; i++)
                                        InputAltered[SpeciesOffset + batchStart + i] += CurrentSearchValues[i][idelta].X;

                                    SetDefocusFromVector(InputAltered, this, true);

                                    for (int p = 0; p < CurBatch; p++)
                                        CoordinatesFrame[p].Z = Particles[batchStart + p].GetSplineCoordinateZ().Interp(DoseInterpolationSteps[f]);

                                    ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                    float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);
                                    for (int p = 0; p < CurBatch; p++)
                                    {
                                        Defoci[p].X = ImageCoords[p].Z;
                                        Defoci[p].Y = Astigmatism[p].X;
                                        Defoci[p].Z = Astigmatism[p].Y;
                                    }

                                    for (int iewald = 0; iewald < (species.DoEwald ? 2 : 1); iewald++)
                                    {
                                        if (species.DoEwald)
                                        {
                                            GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, EwaldReverse[iewald], ExtractedCTF, true);

                                            GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                                ExtractedCTF.GetDevice(Intent.Read),
                                                ExtractedFT.GetDevice(Intent.Write),
                                                ExtractedCTF.ElementsComplex,
                                                1);
                                        }
                                        else
                                        {
                                            GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                            ParticleDiameterPix / 2f * 1.3f,
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
                                            ExtractedCropped.GetDevice(Intent.Write),
                                            new int3(SizeRefine).Slice(),
                                            new int3(RelevantSizes[f]).Slice(),
                                            (uint)CurBatch);

                                        GPU.MultiParticleDiff(EwaldResults[iewald],
                                            new IntPtr[] { ExtractedCropped.GetDevice(Intent.Read) },
                                            SizeRefine,
                                            new[] { RelevantSizes[f] },
                                            new float[CurBatch * 2],
                                            Helper.ToInterleaved(ImageAngles),
                                            MagnificationCorrection.ToVec(),
                                            SpeciesCTFWeights[species].GetDeviceSlice(f, Intent.Read),
                                            PhaseCorrectionAll.GetDeviceSlice(f, Intent.Read),
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
                                        CurrentSearchValues[i][idelta] += new float4(0,
                                            ResultP[i * 3 + 0] + ResultQ[i * 3 + 0],
                                            ResultP[i * 3 + 1] + ResultQ[i * 3 + 1],
                                            ResultP[i * 3 + 2] + ResultQ[i * 3 + 2]);
                                }
                            }

                            GridSearchDelta /= 2;
                            for (int i = 0; i < CurBatch; i++)
                            {
                                CurrentSearchValues[i].Sort((a, b) => -((a.Y / Math.Max(1e-20, Math.Sqrt(a.Z * a.W))).CompareTo(b.Y / Math.Max(1e-20, Math.Sqrt(b.Z * b.W)))));
                                AllSearchValues[i].AddRange(CurrentSearchValues[i]);

                                List<float4> NewSearchValues = new List<float4>();
                                for (int j = 0; j < 10; j++)
                                {
                                    NewSearchValues.Add(new float4(CurrentSearchValues[i][j].X + (float)GridSearchDelta, 0, 0, 0));
                                    NewSearchValues.Add(new float4(CurrentSearchValues[i][j].X - (float)GridSearchDelta, 0, 0, 0));
                                }

                                CurrentSearchValues[i] = NewSearchValues;
                            }
                        }

                        for (int i = 0; i < CurBatch; i++)
                        {
                            AllSearchValues[i].Sort((a, b) => -((a.Y / Math.Max(1e-10, Math.Sqrt(a.Z * a.W))).CompareTo(b.Y / Math.Max(1e-10, Math.Sqrt(b.Z * b.W)))));
                            input[SpeciesOffset + batchStart + i] += AllSearchValues[i][0].X;
                        }
                    }

                    //SumAll.WriteMRC("d_sumall.mrc", true);

                    CoordsCTF.Dispose();
                    GammaCorrection.Dispose();
                    PhaseCorrectionAll.Dispose();
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

                return input;
            };

            #endregion

            GPU.CheckGPUExceptions();

            BroydenFletcherGoldfarbShanno OptimizerWarp = new BroydenFletcherGoldfarbShanno(InitialParametersWarp.Length, WarpEval, WarpGrad);
            BroydenFletcherGoldfarbShanno OptimizerDefocus = new BroydenFletcherGoldfarbShanno(InitialParametersDefocus.Length, CTFEval, CTFGrad);

            bool NeedReextraction = true;

            SetWarpFromVector(InitialParametersWarp, this, true);
            SetDefocusFromVector(InitialParametersDefocus, this, true);

            for (int ioptim = 0; ioptim < optionsMPA.NIterations; ioptim++)
            {
                foreach (var species in allSpecies)
                    species.CurrentMaxShellRefinement = (int)Math.Round(MathHelper.Lerp(optionsMPA.InitialResolutionPercent / 100f,
                            1f,
                            optionsMPA.NIterations == 1 ? 1 : ((float)ioptim / (optionsMPA.NIterations - 1))) *
                        species.HalfMap1Projector[GPUID].Dims.X / 2);

                foreach (var step in OptimizationStepsWarp)
                {
                    if (NeedReextraction)
                    {
                        progressCallback($"Re-extracting particles for optimization iteration {ioptim + 1}/{optionsMPA.NIterations}");
                        AllocateParticleMemory();
                        ReextractPaddedParticles(false);

                        //WarpEval(InitialParametersWarp);
                        //CTFEval(InitialParametersDefocus);

                        GPU.CheckGPUExceptions();
                    }

                    NeedReextraction = false;

                    progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, " + step.Name);

                    BFGSIterations = step.Iterations;
                    CurrentOptimizationTypeWarp = step.Type;
                    CurrentWeightsDict = SpeciesCTFWeights;

                    OptimizerWarp.Maximize(InitialParametersWarp);

                    OldInput = null;

                    GPU.CheckGPUExceptions();
                }

                if (allSpecies.Any(s => s.ResolutionRefinement < (float)optionsMPA.MinimumCTFRefinementResolution))
                {
                    //ReextractPaddedParticles();
                    //WarpEval(InitialParametersWarp);

                    if (ioptim == 0 && optionsMPA.DoDefocusGridSearch)
                    {
                        FreeParticleMemory();

                        CTFEval(InitialParametersDefocus);

                        progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, defocus grid search");

                        InitialParametersDefocus = DefocusGridSearch(InitialParametersDefocus);

                        NeedReextraction = true;

                        GPU.CheckGPUExceptions();
                    }

                    //CurrentWeightsDict = SpeciesFrameWeights;
                    //ReextractPaddedParticles();
                    //WarpEval(InitialParametersWarp);

                    foreach (var step in OptimizationStepsCTF)
                    {
                        FreeParticleMemory();

                        progressCallback($"Running optimization iteration {ioptim + 1}/{optionsMPA.NIterations}, " + step.Name);

                        BFGSIterations = step.Iterations;
                        CurrentOptimizationTypeCTF = step.Type;
                        CurrentWeightsDict = SpeciesCTFWeights;

                        OptimizerDefocus.Maximize(InitialParametersDefocus);

                        OldInput = null;
                        NeedReextraction = true;

                        GPU.CheckGPUExceptions();
                    }
                }
            }

            FreeParticleMemory();

            SetWarpFromVector(InitialParametersWarp, this, true);
            SetDefocusFromVector(InitialParametersDefocus, this, true);

            #region Compute FSC between refs and particles to estimate frame and micrograph weights

            if (true)
            {
                progressCallback($"Calculating FRC between projections and particles for weight optimization");

                int FSCLength = 128;
                Image FSC = new Image(new int3(FSCLength, FSCLength, RedNFrames * 3), true);
                Image FSCPerParticle = new Image(new int3(FSCLength / 2, NParticlesOverall * 3, 1));
                //float[][] FSCData = FSC.GetHost(Intent.ReadWrite);
                //float[][] FSCPerParticleData = FSCPerParticle.GetHost(Intent.ReadWrite);
                Image PhaseResiduals = new Image(new int3(FSCLength, FSCLength, 2), true);

                Star TableOut = new Star(new string[] { "wrpNormCoordinateX", "wrpNormCoordinateY" });

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

                    //Image CorrAB = new Image(new int3(SizeRefine, SizeRefine, NFrames), true);
                    //Image CorrA2 = new Image(new int3(SizeRefine, SizeRefine, NFrames), true);
                    //Image CorrB2 = new Image(new int3(SizeRefine, SizeRefine, NFrames), true);

                    float ScaleFactor = (float)Species.PixelSize * (FSCLength / 2 - 1) /
                                        (float)(Species.ResolutionRefinement / 2 * (SizeRefine / 2 - 1));

                    {
                        int SuperresFactor = SpeciesCTFSuperresFactor[Species];
                        int BatchSize = optionsMPA.BatchSize;
                        BatchSize /= SuperresFactor * SuperresFactor;

                        int SizeRefineSuper = SizeRefine * SpeciesCTFSuperresFactor[Species];
                        int SizeFull = Species.HalfMap1Reconstruction[GPUID].Dims.X;
                        int SizeFullSuper = SizeFull * SpeciesCTFSuperresFactor[Species];
                        float AngPixRefine = Species.ResolutionRefinement / 2;
                        int ParticleDiameterPix = (int)(Species.DiameterAngstrom / AngPixRefine);

                        Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion); // Not SizeFullSuper because CTF creation later adjusts pixel size to AngPixRefine
                        Image CoordsCTFCropped = CTF.GetCTFCoords(SizeRefine, SizeRefine, CTF.Distortion);

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
                                float2 Coords = new float2(CoordinatesMoving[i * NFrames].X, CoordinatesMoving[i * NFrames].Y);
                                Coords /= ImageDimensionsPhysical;
                                TableOut.AddRow(new string[]
                                {
                                    Coords.X.ToString(CultureInfo.InvariantCulture),
                                    Coords.Y.ToString(CultureInfo.InvariantCulture)
                                });
                            }

                            for (int f = 0; f < RedNFrames; f++)
                            {
                                float3[] CoordinatesFrame = new float3[CurBatch];
                                float3[] AnglesFrame = new float3[CurBatch];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                    AnglesFrame[p] = AnglesMoving[p * NFrames + f];
                                }

                                float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                                float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f);
                                float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);

                                float3[] Defoci = new float3[CurBatch];
                                int3[] ExtractOrigins = new int3[CurBatch];
                                float3[] ResidualShifts = new float3[BatchSize];
                                for (int p = 0; p < CurBatch; p++)
                                {
                                    float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                    ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                    ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                    Defoci[p].X = ImageCoords[p].Z;
                                    Defoci[p].Y = Astigmatism[p].X;
                                    Defoci[p].Z = Astigmatism[p].Y;
                                }

                                for (int iewald = 0; iewald < (Species.DoEwald ? 2 : 1); iewald++)
                                {
                                    GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                        Extracted.GetDevice(Intent.Write),
                                        FrameData[f].Dims.Slice(),
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
                                        GetComplexCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, EwaldReverse[iewald], ExtractedCTF, true);

                                        GPU.MultiplyComplexSlicesByComplex(Extracted.GetDevice(Intent.Read),
                                            ExtractedCTF.GetDevice(Intent.Read),
                                            ExtractedFT.GetDevice(Intent.Write),
                                            ExtractedCTF.ElementsComplex,
                                            1);
                                    }
                                    else
                                    {
                                        GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTF, GammaCorrection, f, ExtractedCTF, true);

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
                                        ParticleDiameterPix / 2f * 1.3f,
                                        16 * AngPixExtract / AngPixRefine,
                                        true,
                                        (uint)CurBatch);

                                    GPU.FFT(ExtractedCropped.GetDevice(Intent.Read),
                                        ExtractedCroppedFT.GetDevice(Intent.Write),
                                        new int3(SizeRefine, SizeRefine, 1),
                                        (uint)CurBatch,
                                        PlanForw);

                                    ExtractedCroppedFT.Multiply(1f / (SizeRefine * SizeRefine));

                                    GetCTFsForOneFrame(AngPixRefine, Defoci, ImageCoords, CoordsCTFCropped, null, f, ExtractedCTF, true, true, true);

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
                SpeciesCTFWeights[pair.Key].Dispose();
                SpeciesFrameWeights[pair.Key].Dispose();
                GPU.FreeDevice(SpeciesParticleSubsets[pair.Key]);

                pair.Key.HalfMap1Projector[GPUID].FreeDevice();
                pair.Key.HalfMap2Projector[GPUID].FreeDevice();
            }

            #endregion
        }

        GPU.CheckGPUExceptions();

        #region Update reconstructions with newly aligned particles

        progressCallback($"Extracting and back-projecting particles...");
        if (true)
            foreach (var species in allSpecies)
            {
                if (SpeciesParticles[species].Length == 0)
                    continue;

                Projector[] Reconstructions = { species.HalfMap1Reconstruction[GPUID], species.HalfMap2Reconstruction[GPUID] };

                float AngPixExtract = (float)OptionsDataLoad.BinnedPixelSizeMean;
                int BatchSize = optionsMPA.BatchSize;

                CTF MaxDefocusCTF = CTF.GetCopy();
                float ExpectedResolution = Math.Max((float)dataSource.PixelSizeMean * 2, (float)species.GlobalResolution * 0.8f);
                int ExpectedBoxSize = (int)(species.DiameterAngstrom / (ExpectedResolution / 2)) * 2;
                int MinimumBoxSize = Math.Max(ExpectedBoxSize, MaxDefocusCTF.GetAliasingFreeSize(ExpectedResolution, (float)(species.DiameterAngstrom / AngPixExtract)));
                int CTFSuperresFactor = (int)Math.Ceiling((float)MinimumBoxSize / ExpectedBoxSize);

                float AliasingFreeDiameter = MaxDefocusCTF.GetAliasingFreeSize(ExpectedResolution, (float)(species.DiameterAngstrom / AngPixExtract));

                BatchSize /= CTFSuperresFactor * CTFSuperresFactor;
                BatchSize = 1;

                int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
                int SizeFullSuper = SizeFull * CTFSuperresFactor;

                Image CTFCoords = CTF.GetCTFCoords(SizeFullSuper, SizeFullSuper, CTF.Distortion);
                float2[] CTFCoordsData = CTFCoords.GetHostComplexCopy()[0];
                Image CTFCoordsP = CTF.GetCTFPCoords(SizeFullSuper, SizeFullSuper, CTF.Distortion);
                float2[] CTFCoordsPData = CTFCoordsP.GetHostComplexCopy()[0];
                Image CTFCoordsCropped = CTF.GetCTFCoords(SizeFull, SizeFull, CTF.Distortion);

                Image GammaCorrection = CTF.GetGammaCorrection(AngPixExtract, SizeFullSuper);

                GPU.CheckGPUExceptions();

                float[] PQSigns = new float[CTFCoordsData.Length];
                CTF.PrecomputePQSigns(SizeFullSuper, 2, species.EwaldReverse, CTFCoordsData, CTFCoordsPData, PQSigns);

                GPU.CheckGPUExceptions();

                Image PhaseCorrection = CTF.GetPhaseCorrection(AngPixExtract, SizeFullSuper);

                Image IntermediateMaskAngles = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, 2), true);
                Image IntermediateFTCorr = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                Image IntermediateCTFP = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);

                Image MaskParticle = new Image(new int3(SizeFullSuper, SizeFullSuper, 1));
                MaskParticle.Fill(1);
                MaskParticle.MaskSpherically((float)(species.DiameterAngstrom + 6) / AngPixExtract, 3, false);
                MaskParticle.RemapToFT();

                GPU.CheckGPUExceptions();

                Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize));
                Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
                Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize));
                Image ExtractedCroppedFTp = new Image(new int3(SizeFull, SizeFull, BatchSize), true, true);
                Image ExtractedCroppedFTq = new Image(new int3(SizeFull, SizeFull, BatchSize), true, true);

                Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true);
                Image ExtractedCTFCropped = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize), true);
                Image CTFWeights = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, BatchSize), true);

                GPU.CheckGPUExceptions();

                Image SourceWeight = new Image(IntPtr.Zero, new int3(SizeFull, SizeFull, 1), true);
                {
                    CTF ProtoCTF = CTF.GetCopy();
                    ProtoCTF.Defocus = 0;
                    ProtoCTF.DefocusDelta = 0;
                    ProtoCTF.Cs = 0;
                    ProtoCTF.Amplitude = 1;

                    ProtoCTF.Scale = (decimal)dataSource.RelativeWeight;
                    ProtoCTF.Bfactor = (decimal)dataSource.RelativeBFactor.X;
                    ProtoCTF.BfactorDelta = (decimal)dataSource.RelativeBFactor.Y;
                    ProtoCTF.BfactorAngle = (decimal)dataSource.RelativeBFactor.Z;

                    GPU.CreateCTF(SourceWeight.GetDevice(Intent.Write),
                        CTFCoordsCropped.GetDevice(Intent.Read),
                        IntPtr.Zero,
                        (uint)CTFCoordsCropped.ElementsSliceComplex,
                        new[] { ProtoCTF.ToStruct() },
                        false,
                        1);
                }

                GPU.CheckGPUExceptions();

                int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
                int PlanForw = GPU.CreateFFTPlan(new int3(SizeFull, SizeFull, 1), (uint)BatchSize);

                if (PlanForwSuper <= 0 || PlanBackSuper <= 0 || PlanForw <= 0)
                    throw new Exception("No FFT plans created!");

                Particle[] ParticlesInside = SpeciesParticles[species].Where(p =>
                {
                    float2 Center = new float2(p.Coordinates[0].X, p.Coordinates[0].Y);
                    float2 Rectangle = ImageDimensionsPhysical;
                    float FractionInside = MathHelper.CircleFractionInsideRectangle(Center, AliasingFreeDiameter / 2, new float2(0, 0), Rectangle);

                    return FractionInside > 0.0;
                }).ToArray();

                Console.WriteLine($"{ParticlesInside.Length} particles kept out of {SpeciesParticles[species].Length}, diameter = {AliasingFreeDiameter:F1}, area = {ImageDimensionsPhysical}");
                Console.WriteLine($"Superres factor is {CTFSuperresFactor}");

                Particle[][] SubsetParticles =
                {
                    ParticlesInside.Where(p => p.RandomSubset == 0).ToArray(),
                    ParticlesInside.Where(p => p.RandomSubset == 1).ToArray()
                };

                float4[][] SubsetMags =
                {
                    ParticleMags.Where((m, p) => ParticlesInside[p].RandomSubset == 0).ToArray(),
                    ParticleMags.Where((m, p) => ParticlesInside[p].RandomSubset == 1).ToArray()
                };

                for (int isubset = 0; isubset < 2; isubset++)
                {
                    Particle[] Particles = SubsetParticles[isubset];
                    int NParticles = Particles.Length;

                    for (int batchStart = 0; batchStart < NParticles; batchStart += BatchSize)
                    {
                        int CurBatch = Math.Min(BatchSize, NParticles - batchStart);
                        IEnumerable<Particle> BatchParticles = Particles.Skip(batchStart).Take(CurBatch);
                        float3[] CoordinatesMoving = Helper.Combine(BatchParticles.Select(p => p.GetCoordinateSeries(DoseInterpolationSteps)));
                        float3[] AnglesMoving = Helper.Combine(BatchParticles.Select(p => p.GetAngleSeries(DoseInterpolationSteps)));

                        for (int f = 0; f < RedNFrames; f++)
                        {
                            float3[] CoordinatesFrame = new float3[CurBatch];
                            float3[] AnglesFrame = new float3[CurBatch];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                CoordinatesFrame[p] = CoordinatesMoving[p * NFrames + f];
                                AnglesFrame[p] = AnglesMoving[p * NFrames + f]; // * Helper.ToRad;
                            }

                            float3[] ImageCoords = GetPositionsInOneFrame(CoordinatesFrame, f);
                            float3[] ImageAngles = GetAnglesInOneFrame(CoordinatesFrame, AnglesFrame, f); // AnglesFrame;
                            float2[] Astigmatism = GetAstigmatism(CoordinatesFrame);

                            float3[] Defoci = new float3[CurBatch];
                            int3[] ExtractOrigins = new int3[CurBatch];
                            float3[] ResidualShifts = new float3[BatchSize];
                            for (int p = 0; p < CurBatch; p++)
                            {
                                float ScaledX = ImageCoords[p].X / AngPixExtract, ScaledY = ImageCoords[p].Y / AngPixExtract;
                                ExtractOrigins[p] = new int3((int)ScaledX - SizeFullSuper / 2, (int)ScaledY - SizeFullSuper / 2, 0);
                                ResidualShifts[p] = -new float3(ScaledX - (int)ScaledX - SizeFullSuper / 2, ScaledY - (int)ScaledY - SizeFullSuper / 2, 0);
                                Defoci[p].X = ImageCoords[p].Z;
                                Defoci[p].Y = Astigmatism[p].X;
                                Defoci[p].Z = Astigmatism[p].Y;
                            }

                            #region Image data

                            GPU.Extract(FrameData[f].GetDevice(Intent.Read),
                                Extracted.GetDevice(Intent.Write),
                                FrameData[f].Dims.Slice(),
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

                            CTF[] CTFParams = GetCTFParamsForOneFrame(AngPixExtract, Defoci, ImageCoords, f, false, false, false);

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

                            //CTF.ApplyPandQ(ExtractedFT,
                            //                CTFParams,
                            //                IntermediateFTCorr,
                            //                Extracted,
                            //                ExtractedCropped,
                            //                IntermediateCTFP,
                            //                IntermediateMaskAngles,
                            //                CTFCoordsData,
                            //                CTFCoordsPData,
                            //                MaskParticle,
                            //                PlanForw,
                            //                PlanBackSuper,
                            //                2,
                            //                species.EwaldReverse,
                            //                ExtractedCroppedFTp,
                            //                ExtractedCroppedFTq);

                            GetCTFsForOneFrame(AngPixExtract, Defoci, ImageCoords, CTFCoordsCropped, null, f, CTFWeights, true, true, true);
                            CTFWeights.MultiplySlices(SourceWeight);

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

                            // Try to correct motion-dampened amplitudes
                            //GetCTFsForOneFrame(AngPixExtract, Defoci, ImageCoords, CTFCoordsCropped, f, CTFWeights, true, true, true, true);
                            //GPU.MultiplySlices(ExtractedCTFCropped.GetDevice(Intent.Read),
                            //                    CTFWeights.GetDeviceSlice(f, Intent.Read),
                            //                    ExtractedCTFCropped.GetDevice(Intent.Write),
                            //                    CTFWeights.ElementsSliceReal,
                            //                    (uint)CurBatch);

                            #endregion

                            Reconstructions[isubset].BackProject(ExtractedCroppedFTp, ExtractedCTFCropped, ImageAngles, new Matrix2(SubsetMags[isubset][batchStart]), CTFParams[0].GetEwaldRadius(SizeFull, (float)species.PixelSize));
                            Reconstructions[isubset].BackProject(ExtractedCroppedFTq, ExtractedCTFCropped, ImageAngles, new Matrix2(SubsetMags[isubset][batchStart]), -CTFParams[0].GetEwaldRadius(SizeFull, (float)species.PixelSize));
                        }

                        GPU.CheckGPUExceptions();
                    }
                }

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
                SourceWeight.Dispose();

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

        for (int f = 0; f < FrameData.Length; f++)
            FrameData[f].Dispose();

        #endregion
    }

    public virtual long MultiParticleRefinementCalculateHostMemory(ProcessingOptionsMPARefine optionsMPA,
        Species[] allSpecies,
        DataSource dataSource)
    {
        long Result = 0;

        string DataHash = GetDataHash();
        int GPUID = GPU.GetDevice();

        NFrames = Math.Min(MapHeader.ReadFromFile(DataPath).Dimensions.Z, dataSource.FrameLimit);

        foreach (var species in allSpecies)
        {
            int NParticles = species.GetParticles(DataHash).Length;

            int Size = species.HalfMap1Projector[GPUID].Dims.X;
            int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;

            int[] RelevantSizes = GetRelevantImageSizes(SizeFull, (float)optionsMPA.BFactorWeightingThreshold).Select(v => Math.Min(Size, v)).ToArray();

            long OneSet = Helper.ArrayOfFunction(t => (new int3(RelevantSizes[t]).Slice().ElementsFFT()) * 2 * (long)NParticles * sizeof(float), NFrames).Sum();
            if (species.DoEwald)
                OneSet *= 2;

            Result += OneSet;
        }

        return Result;
    }

    public virtual long MultiParticleRefinementCalculateAvailableDeviceMemory(ProcessingOptionsMPARefine optionsMPA,
        Species[] allSpecies,
        DataSource dataSource)
    {
        string DataHash = GetDataHash();
        int GPUID = GPU.GetDevice();

        long Result = GPU.GetFreeMemory(GPUID);

        NFrames = Math.Min(MapHeader.ReadFromFile(DataPath).Dimensions.Z, dataSource.FrameLimit);

        foreach (var species in allSpecies)
        {
            species.HalfMap1Projector[GPUID].PutTexturesOnDevice();
            species.HalfMap2Projector[GPUID].PutTexturesOnDevice();
        }

        foreach (var species in allSpecies)
        {
            int NParticles = species.GetParticles(DataHash).Length;

            CTF MaxDefocusCTF = CTF.GetCopy();
            int MinimumBoxSize = Math.Max(species.HalfMap1Projector[GPUID].Dims.X, MaxDefocusCTF.GetAliasingFreeSize(species.ResolutionRefinement, (float)(species.DiameterAngstrom / species.ResolutionRefinement * 2)));
            int CTFSuperresFactor = (int)Math.Ceiling((float)MinimumBoxSize / species.HalfMap1Projector[GPUID].Dims.X);

            int SizeRefine = species.HalfMap1Projector[GPUID].Dims.X;
            int SizeRefineSuper = SizeRefine * CTFSuperresFactor;
            int SizeFull = species.HalfMap1Reconstruction[GPUID].Dims.X;
            int SizeFullSuper = species.HalfMap1Reconstruction[GPUID].Dims.X * CTFSuperresFactor;

            int BatchSize = optionsMPA.BatchSize;

            Image CoordsCTF = CTF.GetCTFCoords(SizeRefineSuper, SizeRefineSuper, CTF.Distortion);
            Image PhaseCorrection = CTF.GetPhaseCorrection((float)species.PixelSize, SizeRefineSuper);

            Image Extracted = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
            Image ExtractedFT = new Image(IntPtr.Zero, new int3(SizeFullSuper, SizeFullSuper, BatchSize), true, true);
            Image ExtractedCropped = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize));
            Image ExtractedCroppedFT = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
            Image ExtractedCroppedFTRelevantSize = new Image(IntPtr.Zero, new int3(SizeRefine, SizeRefine, BatchSize), true, true);
            Image ExtractedCTF = new Image(IntPtr.Zero, new int3(SizeRefineSuper, SizeRefineSuper, BatchSize), true);

            int PlanForwSuper = GPU.CreateFFTPlan(new int3(SizeFullSuper, SizeFullSuper, 1), (uint)BatchSize);
            int PlanBackSuper = GPU.CreateIFFTPlan(new int3(SizeRefineSuper, SizeRefineSuper, 1), (uint)BatchSize);
            int PlanForw = GPU.CreateFFTPlan(new int3(SizeRefine, SizeRefine, 1), (uint)BatchSize);

            Result = Math.Min(Result, GPU.GetFreeMemory(GPUID));

            GPU.DestroyFFTPlan(PlanForw);
            GPU.DestroyFFTPlan(PlanBackSuper);
            GPU.DestroyFFTPlan(PlanForwSuper);

            ExtractedCTF.Dispose();
            ExtractedCroppedFTRelevantSize.Dispose();
            ExtractedCroppedFT.Dispose();
            ExtractedCropped.Dispose();
            ExtractedFT.Dispose();
            Extracted.Dispose();

            PhaseCorrection.Dispose();
            CoordsCTF.Dispose();
        }

        Result = Math.Max(0, Result - (1 << 30)); // Subtract 1 GB just in case

        return Result;
    }
}

[Serializable]
public class ProcessingOptionsMPARefine : WarpBase
{
    private int _NIterations = 3;

    [WarpSerializable]
    public int NIterations
    {
        get { return _NIterations; }
        set
        {
            if (value != _NIterations)
            {
                _NIterations = value;
                OnPropertyChanged();
            }
        }
    }

    private decimal _BFactorWeightingThreshold = 0.25M;

    [WarpSerializable]
    public decimal BFactorWeightingThreshold
    {
        get { return _BFactorWeightingThreshold; }
        set
        {
            if (value != _BFactorWeightingThreshold)
            {
                _BFactorWeightingThreshold = value;
                OnPropertyChanged();
            }
        }
    }

    private int _BatchSize = 16;

    [WarpSerializable]
    public int BatchSize
    {
        get { return _BatchSize; }
        set
        {
            if (value != _BatchSize)
            {
                _BatchSize = value;
                OnPropertyChanged();
            }
        }
    }

    private int _InitialResolutionPercent = 80;

    [WarpSerializable]
    public int InitialResolutionPercent
    {
        get { return _InitialResolutionPercent; }
        set
        {
            if (value != _InitialResolutionPercent)
            {
                _InitialResolutionPercent = value;
                OnPropertyChanged();
            }
        }
    }

    private int _MinParticlesPerItem = 10;

    [WarpSerializable]
    public int MinParticlesPerItem
    {
        get { return _MinParticlesPerItem; }
        set
        {
            if (value != _MinParticlesPerItem)
            {
                _MinParticlesPerItem = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _UseHostMemory = true;

    [WarpSerializable]
    public bool UseHostMemory
    {
        get { return _UseHostMemory; }
        set
        {
            if (value != _UseHostMemory)
            {
                _UseHostMemory = value;
                OnPropertyChanged();
            }
        }
    }

    #region Geometry

    private bool _DoImageWarp = true;

    [WarpSerializable]
    public bool DoImageWarp
    {
        get { return _DoImageWarp; }
        set
        {
            if (value != _DoImageWarp)
            {
                _DoImageWarp = value;
                OnPropertyChanged();
            }
        }
    }

    private int _ImageWarpWidth = 3;

    [WarpSerializable]
    public int ImageWarpWidth
    {
        get { return _ImageWarpWidth; }
        set
        {
            if (value != _ImageWarpWidth)
            {
                _ImageWarpWidth = value;
                OnPropertyChanged();
            }
        }
    }

    private int _ImageWarpHeight = 3;

    [WarpSerializable]
    public int ImageWarpHeight
    {
        get { return _ImageWarpHeight; }
        set
        {
            if (value != _ImageWarpHeight)
            {
                _ImageWarpHeight = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoVolumeWarp = false;

    [WarpSerializable]
    public bool DoVolumeWarp
    {
        get { return _DoVolumeWarp; }
        set
        {
            if (value != _DoVolumeWarp)
            {
                _DoVolumeWarp = value;
                OnPropertyChanged();
            }
        }
    }

    private int _VolumeWarpWidth = 3;

    [WarpSerializable]
    public int VolumeWarpWidth
    {
        get { return _VolumeWarpWidth; }
        set
        {
            if (value != _VolumeWarpWidth)
            {
                _VolumeWarpWidth = value;
                OnPropertyChanged();
            }
        }
    }

    private int _VolumeWarpHeight = 3;

    [WarpSerializable]
    public int VolumeWarpHeight
    {
        get { return _VolumeWarpHeight; }
        set
        {
            if (value != _VolumeWarpHeight)
            {
                _VolumeWarpHeight = value;
                OnPropertyChanged();
            }
        }
    }

    private int _VolumeWarpDepth = 2;

    [WarpSerializable]
    public int VolumeWarpDepth
    {
        get { return _VolumeWarpDepth; }
        set
        {
            if (value != _VolumeWarpDepth)
            {
                _VolumeWarpDepth = value;
                OnPropertyChanged();
            }
        }
    }

    private int _VolumeWarpLength = 10;

    [WarpSerializable]
    public int VolumeWarpLength
    {
        get { return _VolumeWarpLength; }
        set
        {
            if (value != _VolumeWarpLength)
            {
                _VolumeWarpLength = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoAxisAngles = false;

    [WarpSerializable]
    public bool DoAxisAngles
    {
        get { return _DoAxisAngles; }
        set
        {
            if (value != _DoAxisAngles)
            {
                _DoAxisAngles = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoParticlePoses = true;

    [WarpSerializable]
    public bool DoParticlePoses
    {
        get { return _DoParticlePoses; }
        set
        {
            if (value != _DoParticlePoses)
            {
                _DoParticlePoses = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoMagnification = false;

    [WarpSerializable]
    public bool DoMagnification
    {
        get { return _DoMagnification; }
        set
        {
            if (value != _DoMagnification)
            {
                _DoMagnification = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoZernike13 = false;

    [WarpSerializable]
    public bool DoZernike13
    {
        get { return _DoZernike13; }
        set
        {
            if (value != _DoZernike13)
            {
                _DoZernike13 = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoZernike5 = false;

    [WarpSerializable]
    public bool DoZernike5
    {
        get { return _DoZernike5; }
        set
        {
            if (value != _DoZernike5)
            {
                _DoZernike5 = value;
                OnPropertyChanged();
            }
        }
    }

    private decimal _GeometryHighPass = 20;

    [WarpSerializable]
    public decimal GeometryHighPass
    {
        get { return _GeometryHighPass; }
        set
        {
            if (value != _GeometryHighPass)
            {
                _GeometryHighPass = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoTiltMovies = false;

    [WarpSerializable]
    public bool DoTiltMovies
    {
        get { return _DoTiltMovies; }
        set
        {
            if (value != _DoTiltMovies)
            {
                _DoTiltMovies = value;
                OnPropertyChanged();
            }
        }
    }

    #endregion

    #region CTF

    private bool _DoDefocus = false;

    [WarpSerializable]
    public bool DoDefocus
    {
        get { return _DoDefocus; }
        set
        {
            if (value != _DoDefocus)
            {
                _DoDefocus = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoAstigmatismDelta = false;

    [WarpSerializable]
    public bool DoAstigmatismDelta
    {
        get { return _DoAstigmatismDelta; }
        set
        {
            if (value != _DoAstigmatismDelta)
            {
                _DoAstigmatismDelta = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoAstigmatismAngle = false;

    [WarpSerializable]
    public bool DoAstigmatismAngle
    {
        get { return _DoAstigmatismAngle; }
        set
        {
            if (value != _DoAstigmatismAngle)
            {
                _DoAstigmatismAngle = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoPhaseShift = false;

    [WarpSerializable]
    public bool DoPhaseShift
    {
        get { return _DoPhaseShift; }
        set
        {
            if (value != _DoPhaseShift)
            {
                _DoPhaseShift = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoCs = false;

    [WarpSerializable]
    public bool DoCs
    {
        get { return _DoCs; }
        set
        {
            if (value != _DoCs)
            {
                _DoCs = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoDoming = false;

    [WarpSerializable]
    public bool DoDoming
    {
        get { return _DoDoming; }
        set
        {
            if (value != _DoDoming)
            {
                _DoDoming = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoZernike2 = false;

    [WarpSerializable]
    public bool DoZernike2
    {
        get { return _DoZernike2; }
        set
        {
            if (value != _DoZernike2)
            {
                _DoZernike2 = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoZernike4 = false;

    [WarpSerializable]
    public bool DoZernike4
    {
        get { return _DoZernike4; }
        set
        {
            if (value != _DoZernike4)
            {
                _DoZernike4 = value;
                OnPropertyChanged();
            }
        }
    }

    private bool _DoDefocusGridSearch = false;

    [WarpSerializable]
    public bool DoDefocusGridSearch
    {
        get { return _DoDefocusGridSearch; }
        set
        {
            if (value != _DoDefocusGridSearch)
            {
                _DoDefocusGridSearch = value;
                OnPropertyChanged();
            }
        }
    }

    private decimal _MinimumCTFRefinementResolution = 7;

    [WarpSerializable]
    public decimal MinimumCTFRefinementResolution
    {
        get { return _MinimumCTFRefinementResolution; }
        set
        {
            if (value != _MinimumCTFRefinementResolution)
            {
                _MinimumCTFRefinementResolution = value;
                OnPropertyChanged();
            }
        }
    }

    private decimal _CTFHighPass = 20;

    [WarpSerializable]
    public decimal CTFHighPass
    {
        get { return _CTFHighPass; }
        set
        {
            if (value != _CTFHighPass)
            {
                _CTFHighPass = value;
                OnPropertyChanged();
            }
        }
    }

    #endregion
}