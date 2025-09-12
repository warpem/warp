using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using Warp.Tools;

namespace Warp;

public partial class TiltSeries
{
    public void ReconstructFull(ProcessingOptionsTomoFullReconstruction options, Func<int3, int, string, bool> progressCallback)
    {
        int GPUID = GPU.GetDevice();

        bool IsCanceled = false;
        string NameWithRes = ToTomogramWithPixelSize(Path, options.BinnedPixelSizeMean);

        Directory.CreateDirectory(ReconstructionDir);

        if (options.DoDeconv)
            Directory.CreateDirectory(ReconstructionDeconvDir);

        if (options.PrepareDenoising)
        {
            Directory.CreateDirectory(ReconstructionOddDir);
            Directory.CreateDirectory(ReconstructionEvenDir);
            Directory.CreateDirectory(ReconstructionCTFDir);
        }

        if (File.Exists(System.IO.Path.Combine(ReconstructionDir, NameWithRes + ".mrc")) && !options.OverwriteFiles)
            return;

        #region Dimensions

        VolumeDimensionsPhysical = options.DimensionsPhysical;

        int3 DimsVolumeCropped = new int3((int)Math.Round(options.DimensionsPhysical.X / (float)options.BinnedPixelSizeMean / 2) * 2,
            (int)Math.Round(options.DimensionsPhysical.Y / (float)options.BinnedPixelSizeMean / 2) * 2,
            (int)Math.Round(options.DimensionsPhysical.Z / (float)options.BinnedPixelSizeMean / 2) * 2);
        int SizeSub = options.SubVolumeSize;
        int SizeSubPadded = (int)(SizeSub * options.SubVolumePadding) * 2;

        #endregion

        #region Establish reconstruction positions

        int3 Grid = (DimsVolumeCropped + SizeSub - 1) / SizeSub;
        List<float3> GridCoords = new List<float3>();
        for (int z = 0; z < Grid.Z; z++)
        for (int y = 0; y < Grid.Y; y++)
        for (int x = 0; x < Grid.X; x++)
            GridCoords.Add(new float3(x * SizeSub + SizeSub / 2,
                y * SizeSub + SizeSub / 2,
                z * SizeSub + SizeSub / 2));

        progressCallback?.Invoke(Grid, 0, "Loading...");

        #endregion

        #region Load and preprocess tilt series

        Movie[] TiltMovies;
        Image[] TiltData, TiltDataOdd, TiltDataEven;
        Image[] TiltMasks;
        LoadMovieData(options, out TiltMovies, out TiltData, options.PrepareDenoising && options.PrepareDenoisingFrames, out TiltDataOdd, out TiltDataEven);
        LoadMovieMasks(options, out TiltMasks);
        Image[][] TiltDataPreprocess = options.PrepareDenoising && options.PrepareDenoisingFrames ? new[] { TiltData, TiltDataEven, TiltDataOdd } : new[] { TiltData };
        for (int z = 0; z < NTilts; z++)
        {
            for (int idata = 0; idata < TiltDataPreprocess.Length; idata++)
            {
                EraseDirt(TiltDataPreprocess[idata][z], TiltMasks[z], noiseScale: 1.0f);
                if (idata == TiltDataPreprocess.Length - 1)
                    TiltMasks[z]?.FreeDevice();

                if (options.Normalize)
                {
                    TiltDataPreprocess[idata][z].SubtractMeanGrid(new int2(1));
                    TiltDataPreprocess[idata][z].MaskRectangularly(new int3(new int2(TiltDataPreprocess[idata][z].Dims) - 32), 16, false);
                    TiltDataPreprocess[idata][z].Bandpass(1f / (SizeSub * (float)options.SubVolumePadding / 2), 1f, false, 0f);

                    GPU.Normalize(TiltDataPreprocess[idata][z].GetDevice(Intent.Read),
                        TiltDataPreprocess[idata][z].GetDevice(Intent.Write),
                        (uint)TiltDataPreprocess[idata][z].ElementsReal,
                        1);
                }

                if (options.Invert)
                    TiltDataPreprocess[idata][z].Multiply(-1f);

                TiltDataPreprocess[idata][z].FreeDevice();
            }
        }

        #endregion

        #region Memory and FFT plan allocation

        Image CTFCoords = CTF.GetCTFCoords(SizeSubPadded, SizeSubPadded);

        Image OutputRecVolume = new Image(DimsVolumeCropped);
        float[][] OutputRec = OutputRecVolume.GetHost(Intent.ReadWrite);
        
        Image OutputRecDeconvVolume = options.DoDeconv ? new Image(DimsVolumeCropped) : null;
        float[][] OutputRecDeconv = OutputRecDeconvVolume?.GetHost(Intent.ReadWrite);
        
        Image OutputRecOddVolume = options.PrepareDenoising ? new Image(DimsVolumeCropped) : null;
        Image OutputRecEvenVolume = options.PrepareDenoising ? new Image(DimsVolumeCropped) : null;
        float[][][] OutputRecHalves = null;
        if (options.PrepareDenoising)
        {
            OutputRecHalves = new[]
            {
                OutputRecOddVolume.GetHost(Intent.ReadWrite),
                OutputRecEvenVolume.GetHost(Intent.ReadWrite)
            };
        }

        int NThreads = 1;

        int[] PlanForw = new int[NThreads], PlanBack = new int[NThreads], PlanForwCTF = new int[NThreads];
        for (int i = 0; i < NThreads; i++)
            Projector.GetPlans(new int3(SizeSubPadded), 1, out PlanForw[i], out PlanBack[i], out PlanForwCTF[i]);
        int[] PlanForwParticle = Helper.ArrayOfFunction(i => GPU.CreateFFTPlan(new int3(SizeSubPadded, SizeSubPadded, 1), (uint)NTilts), NThreads);
        Projector[] Projectors = Helper.ArrayOfFunction(i => new Projector(new int3(SizeSubPadded), 1), NThreads);
        Projector[] Correctors = Helper.ArrayOfFunction(i => new Projector(new int3(SizeSubPadded), 1), NThreads);

        Image[] Subtomo = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded)), NThreads);
        Image[] SubtomoCropped = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSub)), NThreads);

        Image[] Images = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts)), NThreads);
        Image[] ImagesFT = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true, true), NThreads);
        Image[] ImagesFTHalf = options.PrepareDenoising ? Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true, true), NThreads) : null;
        Image[] CTFs = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true), NThreads);
        Image[] CTFsHalf = options.PrepareDenoising ? Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true), NThreads) : null;
        Image[] Samples = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true), NThreads);
        foreach (var samples in Samples)
            samples.Fill(1);
        Image[] SamplesHalf = null;
        if (options.PrepareDenoising)
        {
            SamplesHalf = Helper.ArrayOfFunction(i => new Image(IntPtr.Zero, new int3(SizeSubPadded, SizeSubPadded, NTilts), true), NThreads);
            foreach (var samples in SamplesHalf)
                samples.Fill(1);
        }

        #endregion

        #region Reconstruction

        int NDone = 0;

        Helper.ForCPU(0, GridCoords.Count, NThreads,
            threadID => GPU.SetDevice(GPUID),
            (p, threadID) =>
            {
                if (IsCanceled)
                    return;

                float3 CoordsPhysical = GridCoords[p] * (float)options.BinnedPixelSizeMean;

                GetImagesForOneParticle(options, TiltData, SizeSubPadded, CoordsPhysical, PlanForwParticle[threadID], -1, 8, Images[threadID], ImagesFT[threadID]);
                GetCTFsForOneParticle(options, CoordsPhysical, CTFCoords, null, true, false, false, CTFs[threadID]);

                ImagesFT[threadID].Multiply(CTFs[threadID]); // Weight and phase-flip image FTs

                // We want final amplitudes in reconstruction to remain B-fac weighted. 
                // Thus we will divide B-fac and CTF-weighted data by unweighted CTF, i.e. not by B-fac
                GetCTFsForOneParticle(options, CoordsPhysical, CTFCoords, null, false, false, false, CTFs[threadID]);
                CTFs[threadID].Abs(); // No need for Wiener, just phase flipping

                #region Normal reconstruction

                {
                    Projectors[threadID].Data.Fill(0);
                    Projectors[threadID].Weights.Fill(0);

                    Correctors[threadID].Data.Fill(0);
                    Correctors[threadID].Weights.Fill(0);

                    Projectors[threadID].BackProject(ImagesFT[threadID], CTFs[threadID], GetAngleInAllTilts(CoordsPhysical), MagnificationCorrection);

                    Correctors[threadID].BackProject(ImagesFT[threadID], Samples[threadID], GetAngleInAllTilts(CoordsPhysical), MagnificationCorrection);

                    Correctors[threadID].Weights.Min(1);
                    Projectors[threadID].Data.Multiply(Correctors[threadID].Weights);
                    Projectors[threadID].Weights.Max(0.01f);

                    Projectors[threadID].Reconstruct(Subtomo[threadID].GetDevice(Intent.Write), false, "C1", null, PlanForw[threadID], PlanBack[threadID], PlanForwCTF[threadID], 0);

                    GPU.Pad(Subtomo[threadID].GetDevice(Intent.Read),
                        SubtomoCropped[threadID].GetDevice(Intent.Write),
                        new int3(SizeSubPadded),
                        new int3(SizeSub),
                        1);

                    float[][] SubtomoData = SubtomoCropped[threadID].GetHost(Intent.Read);

                    int3 Origin = new int3(GridCoords[p]) - SizeSub / 2;
                    for (int z = 0; z < SizeSub; z++)
                    {
                        int zVol = Origin.Z + z;
                        if (zVol >= DimsVolumeCropped.Z)
                            continue;

                        for (int y = 0; y < SizeSub; y++)
                        {
                            int yVol = Origin.Y + y;
                            if (yVol >= DimsVolumeCropped.Y)
                                continue;

                            for (int x = 0; x < SizeSub; x++)
                            {
                                int xVol = Origin.X + x;
                                if (xVol >= DimsVolumeCropped.X)
                                    continue;

                                OutputRec[zVol][yVol * DimsVolumeCropped.X + xVol] = SubtomoData[z][y * SizeSub + x];
                            }
                        }
                    }
                }

                #endregion

                #region Odd/even tilt reconstruction

                if (options.PrepareDenoising)
                {
                    for (int ihalf = 0; ihalf < 2; ihalf++)
                    {
                        if (options.PrepareDenoisingTilts)
                        {
                            GPU.CopyDeviceToDevice(ImagesFT[threadID].GetDevice(Intent.Read),
                                ImagesFTHalf[threadID].GetDevice(Intent.Write),
                                ImagesFT[threadID].ElementsReal);
                            GPU.CopyDeviceToDevice(CTFs[threadID].GetDevice(Intent.Read),
                                CTFsHalf[threadID].GetDevice(Intent.Write),
                                CTFs[threadID].ElementsReal);
                            GPU.CopyDeviceToDevice(Samples[threadID].GetDevice(Intent.Read),
                                SamplesHalf[threadID].GetDevice(Intent.Write),
                                Samples[threadID].ElementsReal);
                            ImagesFTHalf[threadID].Multiply(Helper.ArrayOfFunction(i => i % 2 == ihalf ? 1f : 0f, NTilts));
                            CTFsHalf[threadID].Multiply(Helper.ArrayOfFunction(i => i % 2 == ihalf ? 1f : 0f, NTilts));
                            SamplesHalf[threadID].Multiply(Helper.ArrayOfFunction(i => i % 2 == ihalf ? 1f : 0f, NTilts));
                        }
                        else
                        {
                            GetImagesForOneParticle(options, ihalf == 0 ? TiltDataOdd : TiltDataEven, SizeSubPadded, CoordsPhysical, PlanForwParticle[threadID], -1, 8, Images[threadID], ImagesFTHalf[threadID]);
                            GetCTFsForOneParticle(options, CoordsPhysical, CTFCoords, null, true, false, false, CTFsHalf[threadID]);

                            ImagesFTHalf[threadID].Multiply(CTFsHalf[threadID]); // Weight and phase-flip image FTs
                            CTFsHalf[threadID].Abs(); // No need for Wiener, just phase flipping
                        }

                        Projectors[threadID].Data.Fill(0);
                        Projectors[threadID].Weights.Fill(0);

                        Correctors[threadID].Weights.Fill(0);

                        Projectors[threadID].BackProject(ImagesFTHalf[threadID], CTFsHalf[threadID], GetAngleInAllTilts(CoordsPhysical), MagnificationCorrection);

                        Correctors[threadID].BackProject(ImagesFTHalf[threadID], SamplesHalf[threadID], GetAngleInAllTilts(CoordsPhysical), MagnificationCorrection);

                        Correctors[threadID].Weights.Min(1);
                        Projectors[threadID].Data.Multiply(Correctors[threadID].Weights);
                        Projectors[threadID].Weights.Max(0.01f);

                        Projectors[threadID].Reconstruct(Subtomo[threadID].GetDevice(Intent.Write), false, "C1", null, PlanForw[threadID], PlanBack[threadID], PlanForwCTF[threadID], 0);

                        GPU.Pad(Subtomo[threadID].GetDevice(Intent.Read),
                            SubtomoCropped[threadID].GetDevice(Intent.Write),
                            new int3(SizeSubPadded),
                            new int3(SizeSub),
                            1);

                        float[][] SubtomoData = SubtomoCropped[threadID].GetHost(Intent.Read);

                        int3 Origin = new int3(GridCoords[p]) - SizeSub / 2;
                        for (int z = 0; z < SizeSub; z++)
                        {
                            int zVol = Origin.Z + z;
                            if (zVol >= DimsVolumeCropped.Z)
                                continue;

                            for (int y = 0; y < SizeSub; y++)
                            {
                                int yVol = Origin.Y + y;
                                if (yVol >= DimsVolumeCropped.Y)
                                    continue;

                                for (int x = 0; x < SizeSub; x++)
                                {
                                    int xVol = Origin.X + x;
                                    if (xVol >= DimsVolumeCropped.X)
                                        continue;

                                    OutputRecHalves[ihalf][zVol][yVol * DimsVolumeCropped.X + xVol] = SubtomoData[z][y * SizeSub + x];
                                }
                            }
                        }
                    }
                }

                #endregion

                lock(OutputRec)
                    if (progressCallback != null)
                        IsCanceled = progressCallback(Grid, ++NDone, "Reconstructing...");
            }, null);

        // Make 3D CTF for the center of the full tomogram. This can be used to train a deconvolving denoiser.
        // The 3D CTF shouldn't have a missing wedge, so fill everything else with 1s instead of 0s.
        if (options.PrepareDenoising)
        {
            //CTF Center = GetTiltCTF(IndicesSortedAbsoluteAngle[0]);
            //Center.PixelSize = options.BinnedPixelSizeMean;
            //Center.Bfactor = 0;
            //Center.Scale = 1;

            //int Dim = 64;
            //float[] CTF1D = Center.Get1D(Dim / 2, false);
            //Image MapCTF = new Image(new int3(Dim, Dim, Dim), true);
            //{
            //    float[][] ItemData = MapCTF.GetHost(Intent.Write);
            //    Helper.ForEachElementFT(new int3(Dim), (x, y, z, xx, yy, zz, r) =>
            //    {
            //        int r0 = (int)r;
            //        int r1 = r0 + 1;
            //        float v0 = r0 < CTF1D.Length ? CTF1D[r0] : 1;
            //        float v1 = r1 < CTF1D.Length ? CTF1D[r1] : 1;
            //        float v = MathHelper.Lerp(v0, v1, r - r0);

            //        ItemData[z][y * (Dim / 2 + 1) + x] = v;
            //    });
            //}

            //MapCTF.WriteMRC(ReconstructionCTFDir + NameWithRes + ".mrc", (float)options.BinnedPixelSizeMean, true);

            int Dim = 256;

            float3[] ParticlePositions = Helper.ArrayOfConstant(VolumeDimensionsPhysical / 2f, NTilts);
            Image CTFCoords64 = CTF.GetCTFCoords(Dim, Dim);

            Image CTF64 = GetCTFsForOneParticle(options, ParticlePositions, CTFCoords64, null, true, false, false, null);
            Image CTFUnweighted = GetCTFsForOneParticle(options, ParticlePositions, CTFCoords64, null, false, false, false, null);

            CTF64.Multiply(CTFUnweighted);

            Image CTFComplex = new Image(CTF64.Dims, true, true);
            CTFComplex.Fill(new float2(1, 0));
            CTFComplex.Multiply(CTF64);
            CTF64.Dispose();

            CTFUnweighted.Abs();

            Projector Reconstructor = new Projector(new int3(Dim), 1);
            Projector Corrector = new Projector(new int3(Dim), 1);

            Reconstructor.BackProject(CTFComplex, CTFUnweighted, GetAngleInAllTilts(ParticlePositions), MagnificationCorrection);

            CTFUnweighted.Fill(1);
            Corrector.BackProject(CTFComplex, CTFUnweighted, GetAngleInAllTilts(ParticlePositions), MagnificationCorrection);

            Corrector.Weights.Min(1);
            Reconstructor.Data.Multiply(Corrector.Weights);
            Corrector.Dispose();

            CTFComplex.Dispose();
            CTFUnweighted.Dispose();

            Reconstructor.Weights.Max(0.02f);

            Image CTF3D = Reconstructor.Reconstruct(true, "C1", null, 0, 0, 0, 0);
            Reconstructor.Dispose();

            CTF3D.WriteMRC16b(System.IO.Path.Combine(ReconstructionCTFDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
            CTF3D.Dispose();
            CTFCoords64.Dispose();
        }

        #region Teardown

        for (int i = 0; i < NThreads; i++)
        {
            GPU.DestroyFFTPlan(PlanForw[i]);
            GPU.DestroyFFTPlan(PlanBack[i]);
            GPU.DestroyFFTPlan(PlanForwCTF[i]);
            GPU.DestroyFFTPlan(PlanForwParticle[i]);
            Projectors[i].Dispose();
            Correctors[i].Dispose();
            Subtomo[i].Dispose();
            SubtomoCropped[i].Dispose();
            Images[i].Dispose();
            ImagesFT[i].Dispose();
            CTFs[i].Dispose();
            Samples[i].Dispose();
            if (options.PrepareDenoising)
            {
                ImagesFTHalf[i].Dispose();
                CTFsHalf[i].Dispose();
                SamplesHalf[i].Dispose();
            }
        }

        CTFCoords.Dispose();
        foreach (var image in TiltData)
            image.FreeDevice();
        foreach (var tiltMask in TiltMasks)
            tiltMask?.FreeDevice();

        #endregion

        if (IsCanceled)
            return;

        if (options.DoDeconv)
        {
            IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Deconvolving...");

            {
                Image FullRec = new Image(OutputRec, DimsVolumeCropped);

                Image FullRecFT = FullRec.AsFFT_CPU(true);
                FullRec.Dispose();

                CTF SubtomoCTF = CTF.GetCopy();
                SubtomoCTF.Defocus = (decimal)GetTiltDefocus(NTilts / 2);
                SubtomoCTF.PixelSize = options.BinnedPixelSizeMean;

                GPU.DeconvolveCTF(FullRecFT.GetDevice(Intent.Read),
                    FullRecFT.GetDevice(Intent.Write),
                    FullRecFT.Dims,
                    SubtomoCTF.ToStruct(),
                    (float)options.DeconvStrength,
                    (float)options.DeconvFalloff,
                    (float)(options.BinnedPixelSizeMean * 2 / options.DeconvHighpass));

                Image FullRecDeconv = FullRecFT.AsIFFT_CPU(true);
                FullRecFT.Dispose();

                var newData = FullRecDeconv.GetHost(Intent.Read);
                for (int i = 0; i < newData.Length; i++)
                    Array.Copy(newData[i], 0, OutputRecDeconv[i], 0, newData[i].Length);
                
                FullRecDeconv.Dispose();
            }

            if (options.PrepareDenoising)
            {
                for (int ihalf = 0; ihalf < 2; ihalf++)
                {
                    Image FullRec = new Image(OutputRecHalves[ihalf], DimsVolumeCropped);

                    Image FullRecFT = FullRec.AsFFT_CPU(true);
                    FullRec.Dispose();

                    CTF SubtomoCTF = CTF.GetCopy();
                    SubtomoCTF.Defocus = (decimal)GetTiltDefocus(NTilts / 2);
                    SubtomoCTF.PixelSize = options.BinnedPixelSizeMean;

                    GPU.DeconvolveCTF(FullRecFT.GetDevice(Intent.Read),
                        FullRecFT.GetDevice(Intent.Write),
                        FullRecFT.Dims,
                        SubtomoCTF.ToStruct(),
                        (float)options.DeconvStrength,
                        (float)options.DeconvFalloff,
                        (float)(options.BinnedPixelSizeMean * 2 / options.DeconvHighpass));

                    Image FullRecDeconv = FullRecFT.AsIFFT_CPU(true);
                    FullRecFT.Dispose();

                    var newData = FullRecDeconv.GetHost(Intent.Read);
                    for (int i = 0; i < newData.Length; i++)
                        Array.Copy(newData[i], 0, OutputRecHalves[ihalf][i], 0, newData[i].Length);
                    
                    FullRecDeconv.Dispose();
                }
            }
        }

        if (options.KeepOnlyFullVoxels)
        {
            IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Trimming...");

            float BinnedAngPix = (float)options.BinnedPixelSizeMean;

            Parallel.For(0, DimsVolumeCropped.Z, z =>
            {
                float3[] VolumePositions = new float3[DimsVolumeCropped.ElementsSlice()];
                for (int y = 0; y < DimsVolumeCropped.Y; y++)
                for (int x = 0; x < DimsVolumeCropped.X; x++)
                    VolumePositions[y * DimsVolumeCropped.X + x] = new float3(x * BinnedAngPix, y * BinnedAngPix, z * BinnedAngPix);

                float3[] ImagePositions = GetPositionInAllTiltsNoLocalWarp(VolumePositions);

                for (int i = 0; i < ImagePositions.Length; i++)
                {
                    int ii = i / NTilts;
                    int t = i % NTilts;

                    if (ImagePositions[i].X < 0 || ImagePositions[i].Y < 0 ||
                        ImagePositions[i].X > ImageDimensionsPhysical.X - BinnedAngPix ||
                        ImagePositions[i].Y > ImageDimensionsPhysical.Y - BinnedAngPix)
                    {
                        OutputRec[z][ii] = 0;
                        if (options.DoDeconv)
                            OutputRecDeconv[z][ii] = 0;
                        if (options.PrepareDenoising)
                        {
                            OutputRecHalves[0][z][ii] = 0;
                            OutputRecHalves[1][z][ii] = 0;
                        }
                    }
                }
            });
        }

        #endregion

        IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Writing...");

        {
            Image OutputFlat = OutputRecVolume.AsSliceXY(OutputRecVolume.Dims.Z / 2);
            float2 MeanStd;
            {
                Image CentralQuarter = OutputFlat.AsPadded(new int2(OutputFlat.Dims) / 2);
                MeanStd = MathHelper.MeanAndStd(CentralQuarter.GetHost(Intent.Read)[0]);
                CentralQuarter.Dispose();
            }
            float FlatMin = MeanStd.X - MeanStd.Y * 3;
            float FlatMax = MeanStd.X + MeanStd.Y * 3;
            OutputFlat.TransformValues(v => (v - FlatMin) / (FlatMax - FlatMin) * 255);

            OutputFlat.WritePNG(System.IO.Path.Combine(ReconstructionDir, NameWithRes + ".png"));
            OutputFlat.Dispose();
        }
        OutputRecVolume.WriteMRC16b(System.IO.Path.Combine(ReconstructionDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
        OutputRecVolume.Dispose();

        if (options.DoDeconv)
        {
            OutputRecDeconvVolume.WriteMRC16b(System.IO.Path.Combine(ReconstructionDeconvDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
            OutputRecDeconvVolume.Dispose();
        }

        if (options.PrepareDenoising)
        {
            OutputRecOddVolume.WriteMRC16b(System.IO.Path.Combine(ReconstructionOddDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
            OutputRecOddVolume.Dispose();

            OutputRecEvenVolume.WriteMRC16b(System.IO.Path.Combine(ReconstructionEvenDir, NameWithRes + ".mrc"), (float)options.BinnedPixelSizeMean, true);
            OutputRecEvenVolume.Dispose();
        }

        IsCanceled = progressCallback(Grid, (int)Grid.Elements(), "Done.");
    }
}

[Serializable]
public class ProcessingOptionsTomoFullReconstruction : TomoProcessingOptionsBase
{
    [WarpSerializable] public bool OverwriteFiles { get; set; }
    [WarpSerializable] public bool Invert { get; set; }
    [WarpSerializable] public bool Normalize { get; set; }
    [WarpSerializable] public bool DoDeconv { get; set; }
    [WarpSerializable] public decimal DeconvStrength { get; set; }
    [WarpSerializable] public decimal DeconvFalloff { get; set; }
    [WarpSerializable] public decimal DeconvHighpass { get; set; }
    [WarpSerializable] public int SubVolumeSize { get; set; }
    [WarpSerializable] public decimal SubVolumePadding { get; set; }
    [WarpSerializable] public bool PrepareDenoising { get; set; }
    [WarpSerializable] public bool PrepareDenoisingFrames { get; set; }
    [WarpSerializable] public bool PrepareDenoisingTilts { get; set; }
    [WarpSerializable] public bool KeepOnlyFullVoxels { get; set; }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(null, obj)) return false;
        if (ReferenceEquals(this, obj)) return true;
        if (obj.GetType() != this.GetType()) return false;
        return Equals((ProcessingOptionsTomoFullReconstruction)obj);
    }

    protected bool Equals(ProcessingOptionsTomoFullReconstruction other)
    {
        return base.Equals(other) &&
               Invert == other.Invert &&
               Normalize == other.Normalize &&
               DoDeconv == other.DoDeconv &&
               DeconvStrength == other.DeconvStrength &&
               DeconvFalloff == other.DeconvFalloff &&
               DeconvHighpass == other.DeconvHighpass &&
               SubVolumeSize == other.SubVolumeSize &&
               SubVolumePadding == other.SubVolumePadding &&
               PrepareDenoising == other.PrepareDenoising &&
               PrepareDenoisingFrames == other.PrepareDenoisingFrames &&
               PrepareDenoisingTilts == other.PrepareDenoisingTilts &&
               KeepOnlyFullVoxels == other.KeepOnlyFullVoxels;
    }

    public static bool operator ==(ProcessingOptionsTomoFullReconstruction left, ProcessingOptionsTomoFullReconstruction right)
    {
        return Equals(left, right);
    }

    public static bool operator !=(ProcessingOptionsTomoFullReconstruction left, ProcessingOptionsTomoFullReconstruction right)
    {
        return !Equals(left, right);
    }
}