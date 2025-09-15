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
    public void PeakAlign(ProcessingOptionsTomoPeakAlign options, Image template, float3[] positions, float3[] angles)
    {
        #region Dimensions

        VolumeDimensionsPhysical = options.DimensionsPhysical;

        int SizeRegion = (int)(template.Dims.X * (float)options.TemplatePixel / (float)options.BinnedPixelSizeMean / 2 + 1) * 2;

        int NParticles = positions.Length / NTilts;

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

        int2 DimsImage = new int2(TiltData[0].Dims);

        #endregion

        #region Memory and FFT plan allocation

        int PlanForwParticles = GPU.CreateFFTPlan(new int3(SizeRegion, SizeRegion, 1), (uint)NParticles);
        int PlanBackParticles = GPU.CreateIFFTPlan(new int3(SizeRegion, SizeRegion, 1), (uint)NParticles);

        Image Images = new(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NTilts));
        Image ImagesFT = new(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles), true, true);

        Image CTFCoords = CTF.GetCTFCoords(SizeRegion, SizeRegion, Matrix2.Identity());
        Image CTFs = new Image(IntPtr.Zero, new int3(SizeRegion, SizeRegion, NParticles), true);

        #endregion

        #region Projector

        if (template.Dims.X != SizeRegion)
            template = template.AsScaled(new int3(SizeRegion)).AndDisposeParent();

        Projector Projector = new Projector(template, 2);

        #endregion

        Image TiltPeaks = new(new int3(SizeRegion, SizeRegion, NTilts));

        for (int t = 0; t < NTilts; t++)
        {
            float3[] ParticlePositions = Enumerable.Range(0, NParticles).Select(p => positions[p * NTilts + t]).ToArray();
            float3[] ParticleAngles = Enumerable.Range(0, NParticles).Select(p => angles[p * NTilts + t]).ToArray();

            float3[] ParticlePositionsInImage = GetPositionsInOneTilt(ParticlePositions, t);
            float3[] ParticleAnglesInImage = GetAnglesInOneTilt(ParticlePositions, ParticleAngles, t);

            ImagesFT = GetParticleImagesFromOneTilt(options, 
                                                    TiltData, 
                                                    t, 
                                                    SizeRegion, 
                                                    ParticlePositions,
                                                    PlanForwParticles, 
                                                    false, 
                                                    Images, 
                                                    ImagesFT);

            Image References = Projector.Project(new int2(SizeRegion), ParticleAnglesInImage);

            GetCTFsForOneTilt((float)options.BinnedPixelSizeMean,
                              ParticlePositionsInImage.Select(v => v.Z).ToArray(),
                              ParticlePositionsInImage,
                              CTFCoords,
                              null,
                              t,
                              CTFs);
            References.Multiply(CTFs);

            ImagesFT.MultiplyConj(References);

            GPU.IFFT(ImagesFT.GetDevice(Intent.Read), Images.GetDevice(Intent.Write), new int3(SizeRegion).Slice(), (uint)NParticles, PlanBackParticles, true);

            using Image Average = Images.AsReducedAlongZ();
            GPU.CopyDeviceToHost(Average.GetDevice(Intent.Read), TiltPeaks.GetHost(Intent.Write)[t], Average.ElementsReal);

            References.Dispose();
        }

        TiltPeaks.WriteMRC($"d_tiltpeaks_{RootName}.mrc", true);

        int SubpixelFactor = 10;
        TiltPeaks = TiltPeaks.AsScaled(new int2(TiltPeaks.Dims) * SubpixelFactor).AndDisposeParent();
        float2[] Corrections = new float2[NTilts];

        for (int t = 0; t < NTilts; t++)
        {
            float[] PeakData = TiltPeaks.GetHost(Intent.Read)[t];
            int MaxValueIndex = 0;
            float MaxValue = float.MinValue;
            for (int i = 0; i < PeakData.Length; i++)
                if (PeakData[i] > MaxValue)
                {
                    MaxValue = PeakData[i];
                    MaxValueIndex = i;
                }

            float2 PeakPos = new(MaxValueIndex % TiltPeaks.Dims.X, MaxValueIndex / TiltPeaks.Dims.X);
            PeakPos -= new float2(TiltPeaks.Dims.X, TiltPeaks.Dims.Y) / 2;

            Corrections[t] = -PeakPos * ((float)options.BinnedPixelSizeMean / SubpixelFactor);
        }

        if (GridMovementX != null)
            GridMovementX = GridMovementX.Resize(new int3(1, 1, NTilts));
        else
            GridMovementX = new CubicGrid(new int3(1, 1, NTilts));

        if (GridMovementY != null)
            GridMovementY = GridMovementY.Resize(new int3(1, 1, NTilts));
        else
            GridMovementY = new CubicGrid(new int3(1, 1, NTilts));

        GridMovementX = new CubicGrid(new int3(1, 1, NTilts), Corrections.Select((v, i) => v.X + GridMovementX.Values[i]).ToArray());
        GridMovementY = new CubicGrid(new int3(1, 1, NTilts), Corrections.Select((v, i) => v.Y + GridMovementY.Values[i]).ToArray());

        #region Teardown

        TiltPeaks.Dispose();

        Projector.Dispose();

        Images.Dispose();
        ImagesFT.Dispose();
        CTFs.Dispose();
        CTFCoords.Dispose();

        GPU.DestroyFFTPlan(PlanForwParticles);
        GPU.DestroyFFTPlan(PlanBackParticles);

        foreach (var img in TiltData)
            img.Dispose();
        foreach (var img in TiltMasks)
            img?.Dispose();

        #endregion
    }
}

[Serializable]
public class ProcessingOptionsTomoPeakAlign : TomoProcessingOptionsBase
{
    [WarpSerializable] public bool Invert { get; set; }
    [WarpSerializable] public bool Normalize { get; set; }
    [WarpSerializable] public decimal TemplatePixel { get; set; }
    [WarpSerializable] public decimal TemplateDiameter { get; set; }
    [WarpSerializable] public bool WhitenSpectrum { get; set; }
    [WarpSerializable] public decimal Lowpass { get; set; }
    [WarpSerializable] public decimal LowpassSigma { get; set; }
}