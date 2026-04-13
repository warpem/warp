using Accord.Math.Optimization;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Warp.Tools;
using ZLinq;

namespace Warp;

public partial class TiltSeries
{
    public void AddToReconstruction(ProcessingOptionsTomoAddToReconstruction options, Projector[] reconstructions, float3[][] positions, float3[][] angles)
    {
        #region Dimensions

        VolumeDimensionsPhysical = options.DimensionsPhysical;

        int SizeRegion = options.BoxSize;

        #endregion

        #region Load and preprocess data

        Image[] TiltData;
        Image[] TiltMasks;
        LoadMovieData(options, out _, out TiltData, false, out _, out _);
        LoadMovieMasks(options, out TiltMasks);
        for (int z = 0; z < NTilts; z++)
        {
            EraseDirt(TiltData[z], TiltMasks[z]);
            TiltMasks[z]?.FreeDevice();

            TiltData[z].SubtractMeanGrid(new int2(1));
            TiltData[z] = TiltData[z].AsPaddedClampedSoft(new int2(TiltData[z].Dims) * 2, 32).AndDisposeParent();
            TiltData[z].MaskRectangularly((TiltData[z].Dims / 2).Slice(), MathF.Min(TiltData[z].Dims.X / 4, TiltData[z].Dims.Y / 4), false);
            //TiltData[z].WriteMRC("d_tiltdata.mrc", true);
            TiltData[z].Bandpass(1f / (SizeRegion / 2), 1f, false, 1f / (SizeRegion / 2));
            TiltData[z] = TiltData[z].AsPadded(new int2(TiltData[z].Dims) / 2).AndDisposeParent();
            //TiltData[z].WriteMRC("d_tiltdatabp.mrc", true);

            if (options.Invert)
                TiltData[z].Multiply(-1f);

            if (options.Normalize)
                TiltData[z].Normalize();
        }
        GPU.CheckGPUExceptions();

        int2 DimsImage = new int2(TiltData[0].Dims);

        #endregion

        #region Memory and FFT plan allocation

        int PlanForwParticles = GPU.CreateFFTPlan(new int3(SizeRegion, SizeRegion, 1), (uint)options.BatchSize);

        Image Images = new(IntPtr.Zero, new int3(SizeRegion, SizeRegion, options.BatchSize));
        Image ImagesFT = new(IntPtr.Zero, new int3(SizeRegion, SizeRegion, options.BatchSize), true, true);

        Image CTFCoords = CTF.GetCTFCoords(SizeRegion, SizeRegion, Matrix2.Identity());
        Image CTFs = new Image(IntPtr.Zero, new int3(SizeRegion, SizeRegion, options.BatchSize), true);
        Image CTFsUnweighted = new Image(IntPtr.Zero, new int3(SizeRegion, SizeRegion, options.BatchSize), true);
        GPU.CheckGPUExceptions();

        #endregion

        for (int irec = 0; irec < reconstructions.Length; irec++)
        {
            int NParticles = positions[irec].Length / NTilts;

            for (int b = 0; b < NParticles; b+= options.BatchSize)
            {
                int CurBatch = Math.Min(options.BatchSize, NParticles - b);

                int[] ParticleIndices = Enumerable.Range(b, CurBatch).ToArray();
                if (CurBatch < options.BatchSize)
                    ParticleIndices = Helper.Combine(ParticleIndices, new int[options.BatchSize - CurBatch]);

                int[] RelevantTilts = options.LimitFirstNTilts > 0 ? 
                                        IndicesSortedDose.Take(options.LimitFirstNTilts).ToArray() : 
                                        Enumerable.Range(0, NTilts).ToArray();

                foreach (var t in RelevantTilts)
                {
                    if (!UseTilt[t])
                        continue;

                    float3[] ParticlePositions = ParticleIndices.Select(p => positions[irec][p * NTilts + t]).ToArray();
                    float3[] ParticleAngles = ParticleIndices.Select(p => angles[irec][p * NTilts + t]).ToArray();

                    float3[] ParticlePositionsInImage = GetPositionsInOneTilt(ParticlePositions, t);
                    float3[] ParticleAnglesInImage = GetAnglesInOneTilt(ParticlePositions, ParticleAngles, t);

                    ImagesFT = GetParticleImagesFromOneTilt(options,
                                                            TiltData,
                                                            t,
                                                            SizeRegion,
                                                            ParticlePositions,
                                                            PlanForwParticles,
                                                            true,
                                                            Images,
                                                            ImagesFT);
                    GPU.CheckGPUExceptions();

                    GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                      ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                                      ParticlePositionsInImage,
                                      CTFCoords,
                                      null,
                                      t,
                                      CTFs,
                                      weighted: true);
                    GPU.CheckGPUExceptions();

                    GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                                      ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                                      ParticlePositionsInImage,
                                      CTFCoords,
                                      null,
                                      t,
                                      CTFsUnweighted,
                                      weighted: false);
                    GPU.CheckGPUExceptions();

                    ImagesFT.Multiply(CTFs);
                    CTFs.Multiply(CTFsUnweighted);
                    GPU.CheckGPUExceptions();

                    reconstructions[irec].BackProject(ImagesFT, 
                                                 CTFs, 
                                                 ParticleAnglesInImage.Take(CurBatch).ToArray(),
                                                 MagnificationCorrection);
                    GPU.CheckGPUExceptions();
                }

                GPU.CheckGPUExceptions();
            }

            reconstructions[irec].FreeDevice();
        }

        #region Teardown

        Images.Dispose();
        ImagesFT.Dispose();
        CTFs.Dispose();
        CTFsUnweighted.Dispose();
        CTFCoords.Dispose();

        GPU.DestroyFFTPlan(PlanForwParticles);

        foreach (var img in TiltData)
            img.Dispose();
        foreach (var img in TiltMasks)
            img?.Dispose();

        GPU.CheckGPUExceptions();

        #endregion
    }
}

[Serializable]
public class ProcessingOptionsTomoAddToReconstruction : TomoProcessingOptionsBase
{
    [WarpSerializable] public bool Invert { get; set; }
    [WarpSerializable] public bool Normalize { get; set; }
    [WarpSerializable] public int BoxSize { get; set; }
    [WarpSerializable] public int BatchSize { get; set; }
    [WarpSerializable] public int LimitFirstNTilts { get; set; }
}