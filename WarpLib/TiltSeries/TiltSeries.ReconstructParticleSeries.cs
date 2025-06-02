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
    public void ReconstructParticleSeries(ProcessingOptionsTomoSubReconstruction options, float3[] positions, float3[] angles, string tablePath, out Star tableOut)
    {
        bool IsCanceled = false;

        if (!Directory.Exists(ParticleSeriesDir))
            Directory.CreateDirectory(ParticleSeriesDir);

        #region Dimensions

        VolumeDimensionsPhysical = options.DimensionsPhysical;

        int SizeSub = options.BoxSize;

        #endregion

        #region Load and preprocess tilt series

        Movie[] TiltMovies;
        Image[] TiltData;
        Image[] TiltMasks;
        LoadMovieData(options, out TiltMovies, out TiltData, false, out _, out _);
        LoadMovieMasks(options, out TiltMasks);
        for (int z = 0; z < NTilts; z++)
        {
            EraseDirt(TiltData[z], TiltMasks[z]);
            TiltMasks[z]?.FreeDevice();

            if (options.NormalizeInput)
            {
                TiltData[z].SubtractMeanGrid(new int2(1));
                TiltData[z].Bandpass(1f / SizeSub, 1f, false, 0f);

                GPU.Normalize(TiltData[z].GetDevice(Intent.Read),
                    TiltData[z].GetDevice(Intent.Write),
                    (uint)TiltData[z].ElementsReal,
                    1);
            }
            else
            {
                TiltData[z].Bandpass(1f / SizeSub, 1f, false, 0f);
            }

            if (options.Invert)
                TiltData[z].Multiply(-1f);

            TiltData[z].FreeDevice();

            //TiltData[z].Multiply(TiltMasks[z]);
        }

        #endregion

        #region Memory and FFT plan allocation

        int PlanForwParticle = GPU.CreateFFTPlan(new int3(SizeSub, SizeSub, 1), (uint)NTilts);
        int PlanBackParticle = GPU.CreateIFFTPlan(new int3(SizeSub, SizeSub, 1), (uint)NTilts);

        #endregion

        #region Create STAR table

        tableOut = new Star(new string[]
        {
            "rlnTomoName",
            "rlnTomoParticleId",
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnTomoParticleName",
            "rlnOpticsGroup",
            "rlnImageName",
            "rlnOriginXAngst",
            "rlnOriginYAngst",
            "rlnOriginZAngst",
            "rlnTomoVisibleFrames"
        });

        #endregion

        Random Rand = new Random(Name.GetHashCode());

        CTF[] TiltCTFs = Helper.ArrayOfFunction(t => GetTiltCTF(t), NTilts);
        float PixelSize = (float)options.BinnedPixelSizeMean;

        int NParticles = positions.Length / NTilts;
        List<int> UsedTilts = options.DoLimitDose ? IndicesSortedDose.Take(options.NTilts).ToList() : IndicesSortedDose.ToList();
        UsedTilts.Sort();

        Image Images = new Image(new int3(SizeSub, SizeSub, NTilts));
        Image ImagesFT = new Image(new int3(SizeSub, SizeSub, NTilts), true, true);
        Image CTFs = new Image(new int3(SizeSub, SizeSub, NTilts), true, false);
        Image CTFCoords = CTF.GetCTFCoords(SizeSub, SizeSub);

        Image UsedParticles = new Image(new int3(SizeSub, SizeSub, UsedTilts.Count));
        float[][] UsedParticlesData = UsedParticles.GetHost(Intent.Write);

        Image SumAllParticles = new Image(new int3(SizeSub, SizeSub, NTilts));

        // Needed as long as we can't specify per-tilt B-factors in tomograms.star
        // B-factors are strictly tied to exposure, and aren't B-factors but Niko's formula
        Image RelionWeights = new Image(new int3(SizeSub, SizeSub, NTilts), true, false);
        {
            float[][] WeightsData = RelionWeights.GetHost(Intent.Write);
            float a = 0.245f;
            float b = -1.665f;
            float c = 2.81f;

            for (int t = 0; t < NTilts; t++)
            {
                CTF TiltCTF = GetCTFParamsForOneTilt((float)options.BinnedPixelSizeMean,
                    new[] { 0f },
                    new[] { VolumeDimensionsPhysical * 0.5f },
                    t,
                    true)[0];

                Helper.ForEachElementFT(new int2(SizeSub), (x, y, xx, yy) =>
                {
                    float r = MathF.Sqrt(xx * xx + yy * yy);
                    float k = r / (SizeSub * (float)options.BinnedPixelSizeMean);
                    float d0 = a * MathF.Pow(k, b) + c;

                    WeightsData[t][y * (SizeSub / 2 + 1) + x] = MathF.Exp(-0.5f * Dose[t] / d0) * (float)TiltCTF.Scale;
                });
            }
        }

        for (int p = 0; p < NParticles; p++)
        {
            float3[] ParticlePositions = positions.Skip(p * NTilts).Take(NTilts).ToArray();

            ImagesFT = GetImagesForOneParticle(options, TiltData, SizeSub, ParticlePositions, PlanForwParticle, -1, 0, false, Images, ImagesFT);
            GetCTFsForOneParticle(options, ParticlePositions, CTFCoords, null, false, false, false, CTFs);

            ImagesFT.Multiply(CTFs);
            ImagesFT.Multiply(RelionWeights);

            GPU.IFFT(ImagesFT.GetDevice(Intent.Read),
                Images.GetDevice(Intent.Write),
                ImagesFT.Dims.Slice(),
                (uint)ImagesFT.Dims.Z,
                PlanBackParticle,
                false);

            SumAllParticles.Add(Images);

            float[][] ImagesData = Images.GetHost(Intent.Read);
            for (int i = 0; i < UsedTilts.Count; i++)
                Array.Copy(ImagesData[UsedTilts[i]], 0, UsedParticlesData[i], 0, ImagesData[0].Length);

            bool[] Visibility = GetPositionInAllTilts(positions.Skip(p * NTilts).Take(NTilts).ToArray()).Select(p =>
            {
                return p.X > options.ParticleDiameter / 2 && p.X < ImageDimensionsPhysical.X - options.ParticleDiameter / 2 &&
                       p.Y > options.ParticleDiameter / 2 && p.Y < ImageDimensionsPhysical.Y - options.ParticleDiameter / 2;
            }).ToArray();
            for (int t = 0; t < NTilts; t++)
                Visibility[t] = Visibility[t] && UseTilt[t];
            Visibility = Helper.IndexedSubset(Visibility, UsedTilts.ToArray());


            float3 Position0 = positions[p * NTilts + NTilts / 2] / (float)options.BinnedPixelSizeMean;
            float3 Angle0 = angles[p * NTilts + NTilts / 2];

            string SeriesPath = System.IO.Path.Combine(ParticleSeriesDir, $"{RootName}{options.Suffix}_{options.BinnedPixelSizeMean:F2}A_{(p + 1):D6}.mrcs");
            string SeriesPathRelative = Helper.MakePathRelativeTo(SeriesPath, tablePath);

            tableOut.AddRow(new string[]
            {
                RootName + ".tomostar",
                (p + 1).ToString(),
                Position0.X.ToString("F3", CultureInfo.InvariantCulture),
                Position0.Y.ToString("F3", CultureInfo.InvariantCulture),
                Position0.Z.ToString("F3", CultureInfo.InvariantCulture),
                Angle0.X.ToString("F3", CultureInfo.InvariantCulture),
                Angle0.Y.ToString("F3", CultureInfo.InvariantCulture),
                Angle0.Z.ToString("F3", CultureInfo.InvariantCulture),
                $"{RootName}/{p + 1}",
                $"{RootName}",
                SeriesPathRelative,
                "0.0",
                "0.0",
                "0.0",
                $"[{string.Join(',', Visibility.Select(v => v ? "1" : "0"))}]"
            });

            UsedParticles.WriteMRC16b(SeriesPath, (float)options.BinnedPixelSizeMean, true);

            if (IsCanceled)
                break;
        }

        // Save the average of all particle stacks
        {
            SumAllParticles.Multiply(1f / Math.Max(1, NParticles));

            float[][] SumAllParticlesData = SumAllParticles.GetHost(Intent.Read);
            for (int i = 0; i < UsedTilts.Count; i++)
                Array.Copy(SumAllParticlesData[UsedTilts[i]], 0, UsedParticlesData[i], 0, SumAllParticlesData[0].Length);

            string SumPath = System.IO.Path.Combine(ParticleSeriesDir, $"{RootName}{options.Suffix}_{options.BinnedPixelSizeMean:F2}A_average.mrcs");
            UsedParticles.WriteMRC16b(SumPath, (float)options.BinnedPixelSizeMean, true);
        }

        #region Teardown

        RelionWeights.Dispose();
        SumAllParticles.Dispose();
        UsedParticles.Dispose();
        Images.Dispose();
        ImagesFT.Dispose();
        CTFs.Dispose();
        CTFCoords.Dispose();

        GPU.DestroyFFTPlan(PlanForwParticle);
        GPU.DestroyFFTPlan(PlanBackParticle);

        foreach (var image in TiltData)
            image.FreeDevice();
        foreach (var tiltMask in TiltMasks)
            tiltMask?.FreeDevice();

        #endregion
    }
}